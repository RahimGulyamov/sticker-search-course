import io
import json
import tarfile

import pytest

from sticker_search.common import sha256
from sticker_search.file_bundle import pack_files, unpack_files


def test_roundtrip_shards_and_resume(tmp_path):
    root = tmp_path/'source'; root.mkdir()
    (root/'a.bin').write_bytes(bytes(range(256))*100)
    (root/'nested').mkdir()
    (root/'nested/b.json').write_text('{"test": true}')
    bundle, restored = tmp_path/'bundle', tmp_path/'restored'
    pack_files(root, ['a.bin', 'nested/b.json'], bundle, shard_bytes=32)
    assert len(json.loads((bundle/'bundle.json').read_text())['files']) == 2
    unpack_files(bundle, restored)
    unpack_files(bundle, restored)
    for name in ['a.bin', 'nested/b.json']:
        assert (root/name).read_bytes() == (restored/name).read_bytes()
    (restored/'a.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Refusing'):
        unpack_files(bundle, restored)


def test_reject_traversal_even_with_valid_checksum(tmp_path):
    bundle = tmp_path/'bundle'; bundle.mkdir()
    shard = bundle/'files-00000.tar'
    with tarfile.open(shard, 'w') as tar:
        info = tarfile.TarInfo('../escape'); info.size = 1
        tar.addfile(info, io.BytesIO(b'x'))
    (bundle/'bundle.json').write_text(json.dumps(dict(kind='regular_files',
        files=[dict(file=shard.name, sha256=sha256(shard))],
        members=[dict(path='../escape', bytes=1, sha256='unused')])) )
    with pytest.raises(ValueError, match='Unsafe'):
        unpack_files(bundle, tmp_path/'restored')
    assert not (tmp_path/'escape').exists()


def test_reject_corrupt_archive(tmp_path):
    root = tmp_path/'source'; root.mkdir()
    (root/'x').write_bytes(b'one')
    bundle = tmp_path/'bundle'
    pack_files(root, ['x'], bundle)
    (bundle/'files-00000.tar').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Corrupt'):
        unpack_files(bundle, tmp_path/'restored')
