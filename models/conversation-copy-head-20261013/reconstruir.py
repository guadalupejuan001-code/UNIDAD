"""Reconstruct the experimental own C7 candidate from four verified delta tensors."""
import argparse
import hashlib
import json
from pathlib import Path

PARENT_ID = 'unidad-conversation-bd1975c26297d4665dd95b2ad7d36349c62404a2489b72894d1a67f354a228d2'
PARENT_WEIGHT_SHA = '666527de240a7fc9b082f233699a1c808c6bdd8f929651824ce4975377c7c836'
DELTA_SHA = 'e99fe7f8f73725d25c700191a680885171361adc0139853d9d2cfe69533dc28a'
EXPECTED_WEIGHT_SHA = 'c3d2d09bd4f82efd8c92f649482441e1e40bc29cd192a35ddfc1d56a45df400b'
NAMES = {'copy_query.weight', 'copy_key.weight', 'copy_gate.weight', 'copy_gate.bias'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--base', type=Path, default=root.parent / 'conversation-reference-c7')
    args = parser.parse_args()
    from ia_propia.conversation.artifacts import load_artifact
    from safetensors.torch import load, save
    import torch
    model, _, manifest = load_artifact(args.base)
    if manifest['modelId'] != PARENT_ID or digest((args.base / 'weights.safetensors').read_bytes()) != PARENT_WEIGHT_SHA:
        raise ValueError('La base debe ser exactamente la referencia propia C7 publicada.')
    delta_bytes = (root / 'head_delta.safetensors').read_bytes()
    if digest(delta_bytes) != DELTA_SHA:
        raise ValueError('El delta no coincide con su SHA256.')
    delta = load(delta_bytes)
    tensors = model.state_dict()
    if set(delta) != NAMES:
        raise ValueError('El delta debe contener únicamente los cuatro tensores de copia.')
    for name, value in delta.items():
        if value.dtype != tensors[name].dtype or value.shape != tensors[name].shape or not torch.isfinite(value).all():
            raise ValueError('El delta contiene un Tensor incompatible o no finito.')
        tensors[name] = value
    data = save({name: value.detach().cpu().contiguous().clone() for name, value in tensors.items()})
    if digest(data) != EXPECTED_WEIGHT_SHA:
        raise ValueError('La reconstrucción no produce los bytes del candidato evaluado.')
    target = root / 'weights.safetensors'
    if target.exists():
        if target.is_symlink() or digest(target.read_bytes()) != EXPECTED_WEIGHT_SHA:
            raise ValueError('Ya existe un archivo de pesos distinto; no se reemplazó.')
    else:
        with target.open('xb') as output:
            output.write(data)
    verified, _, candidate = load_artifact(root)
    print(json.dumps({'modelId': candidate['modelId'], 'parameterCount': verified.parameter_count,
                      'weightsSha256': digest(data), 'experimental': True,
                      'productionAccepted': False, 'generalConversationValidated': False}))


if __name__ == '__main__':
    main()
