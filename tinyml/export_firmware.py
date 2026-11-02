"""Export selected model and frontend constants into the fresh PlatformIO project."""
import argparse, json, pathlib, shutil
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent

def array(name,a):
    a=np.asarray(a)
    dims=''.join(f'[{v}]' for v in a.shape)
    def encode(x):
        if x.ndim==1: return '{'+','.join(f'{float(v):.9e}f' for v in x)+'}'
        return '{\n'+',\n'.join(encode(y) for y in x)+'\n}'
    return f'static const float {name}{dims} = '+encode(a)+';\n'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--feature',choices=['mfe','mfcc'])
    args=parser.parse_args()
    kind=args.feature or json.loads((ROOT/'comparison.json').read_text())['selected']
    src=ROOT/'platformio'/'src'; src.mkdir(parents=True,exist_ok=True)
    for file in ['model_data.h','model_data.cpp']:
        shutil.copy2(ROOT/kind/file,src/file)
    m=json.loads((ROOT/kind/'metrics.json').read_text()); f=np.load(ROOT/kind/'frontend.npz')
    header='#pragma once\n'
    header+=f'constexpr bool USE_MFCC={str(kind=="mfcc").lower()};\n'
    header+=f'constexpr int FEATURE_BINS={m["input_shape"][2]}, FEATURE_ROWS={m["input_shape"][1]};\n'
    header+=f'constexpr float DETECTION_THRESHOLD={m["threshold"]:.9e}f;\n'
    for name in ['window','mel','dct','mean','std']: header+=array('k_'+name,f[name])
    (src/'frontend_constants.h').write_text(header)
    shutil.copy2(ROOT/kind/'metrics.json',ROOT/'platformio'/'model_metadata.json')
    print('Exported '+kind)
