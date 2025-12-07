#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from .base_distributor import TypeBackend, TypeFloatingMatmulPrecision
from .protocol import DistributorProtocol, TypeCkptState, TypeStrategy, TypePrecision
from ..xutils import lib_utils

def get_distributor(
        backend: TypeBackend,
        seed: int,
        tracker='tb',
        accelerator='auto',
        devices='auto',
        float32_matmul_precision: TypeFloatingMatmulPrecision = 'highest',
        precision: TypePrecision = '32-true',
        strategy: TypeStrategy = 'auto',
) -> DistributorProtocol:
    if backend == 'none':
        from .dummy_distributor import DummyDistributor
        return DummyDistributor(
            seed,
            tracker=tracker,
            accelerator=accelerator,
            devices=devices,
            float32_matmul_precision=float32_matmul_precision,
            precision=precision,
            strategy=strategy,
        )
    if backend == 'fabric':
        if not lib_utils.import_available('lightning.Fabric'):
            raise RuntimeError(
                'Failed to import lightning.Fabric. '
                'Please install lightning with fabric support.'
            )
        from .fabric_distributor import FabricDistributor
        return FabricDistributor(
            seed,
            tracker=tracker,
            accelerator=accelerator,
            devices=devices,
            float32_matmul_precision=float32_matmul_precision,
            precision=precision,
            strategy=strategy,
        )
    raise NotImplementedError('backend')