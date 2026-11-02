"""Fresh, reproducible MFCC/MFE training. See README.md for evaluation limits."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse, hashlib, json, math, pathlib, re
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.fft import dct
import tensorflow as tf
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

ROOT = pathlib.Path(__file__).resolve().parent
LABELS = ['hello_vaani', 'noise', 'silence', 'speech']
RATE, SAMPLES, FRAME, HOP, FFT, BINS = 16000, 32000, 400, 320, 512, 32
SEED = 20260909
tf.keras.utils.set_random_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf8')

def mel_matrix():
    hz = np.linspace(0, RATE/2, FFT//2+1)
    mel = lambda f: 2595*np.log10(1+f/700)
    edges = 700*(10**(np.linspace(mel(80), mel(7600), BINS+2)/2595)-1)
    return np.maximum(0, np.minimum((hz[:,None]-edges[:-2])/(edges[1:-1]-edges[:-2]),
                         (edges[2:]-hz[:,None])/(edges[2:]-edges[1:-1]))).astype('float32')

MEL = mel_matrix()
WINDOW = np.hanning(FRAME+1)[:-1].astype('float32')
DCT = dct(np.eye(BINS), type=2, norm='ortho', axis=0)[:13].astype('float32')

def read_audio(p):
    x, sr = sf.read(p, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if not np.isfinite(x).all(): raise ValueError(f'Nonfinite audio: {p}')
    if sr != RATE:
        g = math.gcd(sr, RATE)
        x = resample_poly(x, RATE//g, sr//g).astype('float32')
    return x

def fit_audio(x, keyword=False, rng=None):
    # Repeat short negative clips instead of giving only negatives zero-padding.
    if len(x) < SAMPLES and not keyword:
        x = np.tile(x, math.ceil(SAMPLES/max(1,len(x))))
    if len(x)>SAMPLES:
        start = int(rng.integers(len(x)-SAMPLES+1)) if rng is not None else (len(x)-SAMPLES)//2
        x = x[start:start+SAMPLES]
    if len(x)<SAMPLES:
        gap = SAMPLES-len(x)
        left = int(rng.integers(gap+1)) if rng is not None else gap//2
        x = np.pad(x, (left,gap-left))
    return x.astype('float32')

def extract(x):
    frames = np.lib.stride_tricks.sliding_window_view(x, FRAME)[::HOP]
    power = (np.abs(np.fft.rfft(frames*WINDOW, n=FFT))**2 / FFT).astype('float32')
    mfe = np.log(np.maximum(power@MEL, 1e-10)).astype('float32')
    return mfe, (mfe@DCT.T).astype('float32')

def prepare(dataset, cache):
    rng = np.random.default_rng(SEED)
    records, seen, excluded, duplicates = [], {}, 0, 0
    for p in sorted(dataset.rglob('*.wav')):
        rel = p.relative_to(dataset)
        if 'keyword_augmented' in rel.parts:
            excluded += 1; continue
        label = {'hello vaani':0,'noise':1,'silence':2,'speech':3}.get(rel.parts[0])
        if label is None: continue
        x = read_audio(p)
        if not len(x): raise ValueError(f'Empty recording: {p}')
        digest = hashlib.sha256(x.tobytes()).hexdigest()
        if digest in seen:
            if seen[digest] != label: raise ValueError(f'Conflicting duplicate label: {p}')
            duplicates += 1; continue
        seen[digest] = label
        records.append(dict(path=str(p),label=label,hash=digest,synthetic='keyword_tts' in rel.parts,
                            seconds=len(x)/RATE,source=str(rel.parent)))
    for label in range(4):
        ids = [i for i,r in enumerate(records) if r['label']==label and not r['synthetic']]
        rng.shuffle(ids)
        if len(ids)<10: raise ValueError(f'Too few unique real examples for {LABELS[label]}: {len(ids)}')
        a,b = int(.7*len(ids)),int(.85*len(ids))
        for j,i in enumerate(ids): records[i]['split'] = 'train' if j<a else 'validation' if j<b else 'test'
    for r in records:
        if r['synthetic']: r['split']='train'
    dump(ROOT/'manifest.json', records)
    audit = dict(unique_files=len(records),excluded_untraceable_augmented=excluded,exact_duplicates_removed=duplicates,
                 counts={s:{LABELS[l]:sum(r['split']==s and r['label']==l for r in records) for l in range(4)}
                         for s in ['train','validation','test']},
                 limitations=['Speaker/session identities unavailable: split is recording-disjoint, not proven speaker-disjoint.',
                 'Unknown near-duplicates may remain; exact decoded audio duplicates removed.',
                 'One-second negatives are repeated to fill two seconds; continuous real microphone validation is still required.'])
    dump(ROOT/'dataset_audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)
    noises=[fit_audio(read_audio(r['path'])) for r in records if r['label']==1 and r['split']=='train']
    data={s: [[],[],[]] for s in ['train','validation','test']}
    for i,r in enumerate(records):
        raw=read_audio(r['path']); s=r['split']; l=r['label']
        repetitions=5 if s=='train' and l==0 and not r['synthetic'] else (2 if s=='train' else 1)
        for a in range(repetitions):
            x=fit_audio(raw,l==0,rng if s=='train' else None)
            if a:
                shift=int(rng.integers(-2400,2401))
                x=np.roll(x,shift)
                if shift>0: x[:shift]=0
                elif shift<0: x[shift:]=0
                x=x*10**(rng.uniform(-8,5)/20)
                if l!=2 and rng.random()<.8:
                    noise=noises[int(rng.integers(len(noises)))]; rms=np.sqrt(np.mean(x*x)+1e-12)
                    x=x+noise*(rms/(np.sqrt(np.mean(noise*noise))+1e-8))*10**(-rng.uniform(8,30)/20)
                x=np.clip(x,-1,1)
            mfe,mfcc=extract(x)
            data[s][0].append(mfe); data[s][1].append(mfcc); data[s][2].append(l)
        if i%300==0: print(f'Features {i}/{len(records)}',flush=True)
    arrays={f'{s}_{f}':np.asarray(v[j],dtype='int32' if f=='y' else 'float32') for s,v in data.items() for j,f in enumerate(['mfe','mfcc','y'])}
    np.savez_compressed(cache,**arrays)
    return arrays

def metrics(y,p,t):
    pred=p.argmax(1); pos=y==0; detect=p[:,0]>=t
    tp=int(np.sum(detect&pos)); fp=int(np.sum(detect&~pos)); fn=int(np.sum(~detect&pos)); tn=int(np.sum(~detect&~pos))
    return dict(accuracy=float(np.mean(y==pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                confusion_matrix=confusion_matrix(y,pred,labels=range(4)).tolist(),
                keyword_recall=tp/max(tp+fn,1),keyword_precision=tp/max(tp+fp,1),
                negative_clip_false_positive_rate=fp/max(fp+tn,1),tp=tp,fp=fp,fn=fn,tn=tn)

def threshold(y,p):
    candidates=[]
    for t in np.linspace(.05,.99,189):
        m=metrics(y,p,float(t))
        candidates.append((m['negative_clip_false_positive_rate']<=.01,m['keyword_recall']-2*m['negative_clip_false_positive_rate'],float(t)))
    return max(candidates)[2]

def make_model(shape):
    layers=tf.keras.layers
    inp=layers.Input(shape=shape)
    x=layers.Conv2D(12,3,strides=2,padding='same',use_bias=False)(inp)
    x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    for filters in [16,24]:
        x=layers.DepthwiseConv2D(3,strides=2,padding='same',use_bias=False)(x)
        x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
        x=layers.Conv2D(filters,1,use_bias=False)(x)
        x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.Flatten()(x); x=layers.Dropout(.25)(x)
    x=layers.Dense(24,activation='relu')(x)
    return tf.keras.Model(inp,layers.Dense(4,activation='softmax')(x))

def infer(interpreter,x):
    inp=interpreter.get_input_details()[0]; out=interpreter.get_output_details()[0]
    scale,zero=inp['quantization']; oscale,ozero=out['quantization']; p=[]
    for row in x:
        q=np.clip(np.rint(row[None]/scale+zero),-128,127).astype('int8')
        interpreter.set_tensor(inp['index'],q); interpreter.invoke()
        p.append((interpreter.get_tensor(out['index'])[0].astype('float32')-ozero)*oscale)
    return np.asarray(p)

def train(kind,data,epochs,export_only=False):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
    out=ROOT/kind; out.mkdir(exist_ok=True)
    trainx=data['train_'+kind]; mean=trainx.mean(axis=(0,1)); std=np.maximum(trainx.std(axis=(0,1)),.1)
    xs={s:((data[s+'_'+kind]-mean)/std)[...,None].astype('float32') for s in ['train','validation','test']}
    ys={s:data[s+'_y'] for s in xs}
    model=make_model(xs['train'].shape[1:]); model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    counts=np.bincount(ys['train'],minlength=4)
    weights={i:float(len(ys['train'])/(4*n)) for i,n in enumerate(counts)}
    if export_only:
        model=tf.keras.models.load_model(out/'model.keras')
    else:
        history=model.fit(xs['train'],ys['train'],validation_data=(xs['validation'],ys['validation']),
           epochs=epochs,batch_size=64,class_weight=weights,verbose=2,
           callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=3,factor=.5,min_lr=1e-5)])
        model.save(out/'model.keras'); dump(out/'history.json',history.history)
    # SavedModel conversion avoids Keras 3 direct-converter variable issues.
    model.export(str(out/'saved_model'),input_signature=[tf.TensorSpec([1,*xs['train'].shape[1:]],tf.float32)])
    converter=tf.lite.TFLiteConverter.from_saved_model(str(out/'saved_model'))
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    rng=np.random.default_rng(SEED)
    representative=np.concatenate([rng.choice(np.flatnonzero(ys['train']==i),min(80,counts[i]),replace=False) for i in range(4)])
    converter.representative_dataset=lambda: ([xs['train'][i:i+1]] for i in representative)
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type=tf.int8; converter.inference_output_type=tf.int8
    blob=converter.convert(); (out/'model_int8.tflite').write_bytes(blob)
    it=tf.lite.Interpreter(model_content=blob,num_threads=2); it.allocate_tensors()
    val=infer(it,xs['validation']); t=threshold(ys['validation'],val)
    # Model and decision threshold are fixed using validation only, before test evaluation.
    test=infer(it,xs['test']); float_test=model.predict(xs['test'],verbose=0)
    inputinfo=it.get_input_details()[0]; outputinfo=it.get_output_details()[0]
    result=dict(feature=kind,labels=LABELS,parameters=model.count_params(),model_bytes=len(blob),
         input_shape=inputinfo['shape'].tolist(),input_quantization=list(inputinfo['quantization']),
         output_quantization=list(outputinfo['quantization']),threshold=t,
         validation=metrics(ys['validation'],val,t),test=metrics(ys['test'],test,t),
         float_test=metrics(ys['test'],float_test,t),
         largest_tensor_bytes=max(int(np.prod(a['shape']))*np.dtype(a['dtype']).itemsize for a in it.get_tensor_details()),
         ram_note='Largest tensor is not total RAM. TFLM arena, DSP buffers and stack need target measurement.',
         frontend=dict(sample_rate=RATE,samples=SAMPLES,frame_samples=FRAME,hop_samples=HOP,fft=FFT,
         mel_bins=BINS,mfcc_coefficients=13,low_hz=80,high_hz=7600,power_divisor=FFT,log='natural',floor=1e-10,
         window='periodic Hann',preemphasis=False,center=False,dct='type II orthonormal',mean=mean.tolist(),std=std.tolist()))
    dump(out/'metrics.json',result)
    np.savez(out/'frontend.npz',mel=MEL,window=WINDOW,dct=DCT,mean=mean,std=std)
    (out/'model_data.h').write_text('#pragma once\n#include <stdint.h>\nextern const unsigned char g_model[];\nextern const unsigned int g_model_len;\n')
    (out/'model_data.cpp').write_text('#include "model_data.h"\nalignas(16) const unsigned char g_model[] = {\n'+
         ',\n'.join(','.join(str(v) for v in blob[i:i+20]) for i in range(0,len(blob),20))+'\n};\nconst unsigned int g_model_len=sizeof(g_model);\n')
    np.savez(out/'test_vectors.npz',features=xs['test'][:8],probabilities=test[:8],labels=ys['test'][:8])
    print('RESULT '+json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',type=pathlib.Path,required=True)
    ap.add_argument('--epochs',type=int,default=40); ap.add_argument('--features',nargs='+',default=['mfe','mfcc'],choices=['mfe','mfcc'])
    ap.add_argument('--export-only',action='store_true',help='Re-export existing checkpoints, without retraining')
    ap.add_argument('--cache',type=pathlib.Path,default=ROOT/'features.npz'); args=ap.parse_args()
    data=dict(np.load(args.cache)) if args.cache.exists() else prepare(args.dataset,args.cache)
    results=[train(kind,data,args.epochs,args.export_only) for kind in args.features]
    best=max(results,key=lambda r:(r['validation']['keyword_recall']-2*r['validation']['negative_clip_false_positive_rate'],r['validation']['balanced_accuracy'],-r['model_bytes']))
    dump(ROOT/'comparison.json',dict(selected=best['feature'],selection='Validation keyword recall minus 2 * negative false-positive rate, then balanced accuracy, then smaller size; test not used.',models=results))
    print('SELECTED '+best['feature'],flush=True)
