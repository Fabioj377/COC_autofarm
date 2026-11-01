import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.utils import player_list_store as store


class PlayerListTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.legacy = self.root / 'legacy.json'
        self.dest = self.data / 'player_list.json'
        for target, value in (('get_user_app_data_dir', self.data),
                              ('_legacy_player_list_paths', [self.legacy])):
            patcher = patch.object(store, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_fresh_install_load_and_save(self):
        self.assertEqual(store.get_player_list_path(), self.dest)
        self.assertFalse(self.dest.exists())
        self.assertEqual(store.load_players(), [])
        store.save_players([store.PlayerEntry('Alice', False)])
        self.assertEqual(store.load_players(), [store.PlayerEntry('Alice', False)])

    def test_migrates_only_existing_legacy_file(self):
        self.legacy.write_text('[{"name": " Alice "}]', encoding='utf-8')
        with patch.object(store, '_legacy_player_list_paths',
                          return_value=[self.root / 'missing.json', self.legacy]):
            self.assertEqual(store.load_players(), [store.PlayerEntry('Alice')])
        self.assertTrue(self.legacy.is_file())

    def test_existing_destination_is_not_overwritten(self):
        self.data.mkdir()
        self.dest.write_text('[{"name": "Current"}]', encoding='utf-8')
        self.legacy.write_text('[{"name": "Old"}]', encoding='utf-8')
        self.assertEqual(store.load_players(), [store.PlayerEntry('Current')])

    def test_invalid_entries_are_skipped(self):
        self.data.mkdir()
        self.dest.write_text(
            '[null, 7, {}, {"name": 15}, {"name": " "}, '
            '{"name": " Alice "}, {"name": "Bob", "enabled": false}]', encoding='utf-8')
        self.assertEqual(store.load_players(),
                         [store.PlayerEntry('Alice'), store.PlayerEntry('Bob', False)])

    def test_invalid_json_shape_and_encoding_return_list(self):
        self.data.mkdir()
        for content in (b'{', b'{}', b'null', b'\xff'):
            with self.subTest(content=content):
                self.dest.write_bytes(content)
                self.assertEqual(store.load_players(), [])

    def test_read_error_returns_list(self):
        self.data.mkdir()
        self.dest.write_text('[]', encoding='utf-8')
        with patch.object(Path, 'read_text', side_effect=PermissionError('denied')):
            self.assertEqual(store.load_players(), [])

    def test_path_or_migration_error_returns_list(self):
        with patch.object(store, 'get_player_list_path', side_effect=OSError('denied')):
            self.assertEqual(store.load_players(), [])
        self.legacy.write_text('[]', encoding='utf-8')
        with patch.object(store.shutil, 'copy2', side_effect=PermissionError('denied')):
            self.assertEqual(store.load_players(), [])


if __name__ == '__main__':
    unittest.main()
