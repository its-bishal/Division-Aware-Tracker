
__NORMALIZER__ ={
    'CLAHE': 'CLAHE',
}

def normalization(name, **kwargs):
    if name not in __NORMALIZER__:
        raise ValueError(f'Normalizer {name} not found. Available normalizers: {list(__NORMALIZER__.keys())}')
    return __NORMALIZER__[name](**kwargs)
