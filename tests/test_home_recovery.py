import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import cv2
import numpy as np

from app.core.bot import Bot
from app.services.vision import NATIVE_SCALE_TEMPLATES, VisionService


class HomeRecoveryTests(unittest.TestCase):
    def setUp(self):
        # No service constructors, Windows access or real input in these tests.
        self.bot = Bot.__new__(Bot)
        self.bot.window, self.bot.input, self.bot.vision = Mock(), Mock(), Mock()
        self.bot.stop_event = Mock()
        self.bot.stop_event.is_set.return_value = False
        self.bot.stop_event.wait.return_value = False
        self.bot._update_config_size = Mock()
        self.bot._battle_rewards = Mock()
        self.bot._battle_rewards.handle.return_value = False
        self.bot._dismiss_okay_or_exit_on_frame = Mock(return_value=False)
        self.bot._reload_first_seen = 123
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.bot.window.screenshot.return_value = self.frame
        self.bot.vision.find_template.return_value = (None, None)
        path = Mock()
        path.exists.return_value = False
        patcher = patch('app.core.bot.get_template_path', return_value=path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def confirm(self, template, expected_log):
        self.bot.vision.find_template.side_effect = (
            lambda frame, name, **kwargs: (100, 100) if name == template else (None, None))
        with self.assertLogs('BotCore', level='INFO') as logs:
            self.bot._home_screen_recovery()
        self.assertEqual(logs.output, ['INFO:BotCore:' + expected_log])
        self.bot.window.screenshot.assert_called_once()
        self.bot.stop_event.wait.assert_not_called()
        self.bot.input.click.assert_not_called()
        self.assertIsNone(self.bot._reload_first_seen)
        names = [c.args[1] for c in self.bot.vision.find_template.call_args_list]
        self.assertEqual(names[-1], template)
        return names

    def test_builder_confirms_immediately(self):
        names = self.confirm('builder.png', 'Recovery: Home Village confirmed via builder')
        self.assertNotIn('gbuilder.png', names)
        self.assertNotIn('attack.png', names)

    def test_goblin_builder_confirms_immediately(self):
        names = self.confirm('gbuilder.png', 'Recovery: Home Village confirmed via goblin builder')
        self.assertIn('builder.png', names)
        self.assertNotIn('attack.png', names)

    def test_attack_confirms_only_after_both_builders_fail(self):
        names = self.confirm('attack.png', 'Recovery: Home Village confirmed via attack button')
        self.assertEqual(names[-3:], ['builder.png', 'gbuilder.png', 'attack.png'])
        self.assertEqual(self.bot.vision.find_template.call_args.kwargs,
                         {'threshold': .8, 'region': (0, 540, 1920, 540)})

    def test_no_match_exhausts_twenty_attempts_with_one_warning(self):
        with self.assertLogs('BotCore', level='INFO') as logs:
            self.bot._home_screen_recovery()
        self.assertEqual(self.bot.window.screenshot.call_count, 20)
        self.assertEqual(self.bot.stop_event.wait.call_args_list, [call(1)] * 20)
        self.assertEqual(logs.output, [
            'WARNING:BotCore:Recovery: Home Village not confirmed after 20 attempts; continuing'])

    def test_stop_before_capture(self):
        self.bot.stop_event.is_set.return_value = True
        with self.assertRaises(InterruptedError):
            self.bot._home_screen_recovery()
        self.bot.window.screenshot.assert_not_called()
        self.bot.input.click.assert_not_called()

    def test_stop_during_wait_has_no_exhaustion_warning(self):
        for attempt in (1, 20):
            with self.subTest(attempt=attempt):
                self.bot.window.screenshot.reset_mock()
                self.bot.stop_event.wait.side_effect = [False] * (attempt - 1) + [True]
                with patch('app.core.bot.logger') as logger:
                    self.bot._home_screen_recovery()
                self.assertEqual(self.bot.window.screenshot.call_count, attempt)
                logger.warning.assert_not_called()

    def test_return_home_then_attack_confirmation_without_extra_wait(self):
        first = np.zeros_like(self.frame)
        self.bot.window.screenshot.side_effect = [first, self.frame]

        def match(frame, name, **kwargs):
            if frame is first and name == 'returnhome.png':
                return (800, 900)
            if frame is self.frame and name == 'attack.png':
                return (100, 950)
            return (None, None)

        self.bot.vision.find_template.side_effect = match
        with self.assertLogs('BotCore', level='INFO') as logs:
            self.bot._home_screen_recovery()
        self.assertEqual(logs.output, [
            'INFO:BotCore:Recovery: clicking Return Home',
            'INFO:BotCore:Recovery: Home Village confirmed via attack button'])
        self.bot.input.click.assert_called_once_with(800, 900, pause=.3)
        self.bot.stop_event.wait.assert_called_once_with(.5)

    def test_recovery_attack_matching_keeps_native_size_and_bottom_roi(self):
        self.assertIn('attack.png', NATIVE_SCALE_TEMPLATES)
        tile = np.random.default_rng(42).integers(0, 256, (84, 152, 3), dtype=np.uint8)

        def match(frame, name, **kwargs):
            if name == 'attack.png':
                return VisionService.find_template(frame, name, **kwargs)
            return (None, None)

        self.bot.vision.find_template.side_effect = match
        cfg = SimpleNamespace(aspect_key='16_9', ref_width=2560, ref_height=1440)
        for top, expected_attempts in ((700, 1), (100, 20)):
            with self.subTest(top=top):
                self.frame.fill(0)
                self.frame[top:top+84, 100:252] = tile
                self.bot.window.screenshot.reset_mock()
                with patch('app.services.vision.Config', return_value=cfg), \
                        patch('app.services.vision.get_template_path', return_value=Path('attack.png')), \
                        patch('app.services.vision.cv2.imread', return_value=tile), \
                        patch('app.services.vision.cv2.resize', wraps=cv2.resize) as resize, \
                        patch('app.core.bot.logger'):
                    self.bot._home_screen_recovery()
                resize.assert_not_called()
                self.assertEqual(self.bot.window.screenshot.call_count, expected_attempts)


if __name__ == '__main__':
    unittest.main()
