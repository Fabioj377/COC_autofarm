import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import cv2
import numpy as np

from app.config import Config
from app.core.bot import Bot
from app.services.vision import NATIVE_SCALE_TEMPLATES, VisionService


class StartupTests(unittest.TestCase):
    def setUp(self):
        # Bypass Bot/WindowService/InputService constructors: no Win32 access.
        self.bot = Bot.__new__(Bot)
        self.bot.window = Mock()
        self.bot.input = Mock()
        self.bot.vision = Mock()
        self.bot.stop_event = Mock()
        self.bot.stop_event.is_set.return_value = False
        self.bot.stop_event.wait.return_value = False
        # Use a detached configuration, without initializing the singleton.
        self.bot.config = object.__new__(Config)
        self.bot.config.aspect_key = '16_9'
        self.bot.config.ref_width, self.bot.config.ref_height = 2560, 1440
        self.bot.config.width, self.bot.config.height = 2560, 1440
        self.bot.config.data = {'empty': [1350, 1410]}
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.bot.window.screenshot.return_value = self.frame

    def test_capture_and_sync_precede_first_click(self):
        order = []
        self.bot.window.screenshot.side_effect = lambda: order.append('capture') or self.frame
        original_sync = self.bot._update_config_size

        def sync(frame):
            order.append('sync')
            original_sync(frame)

        self.bot._update_config_size = Mock(side_effect=sync)
        self.bot.input.click.side_effect = lambda *a, **kw: order.append('click')
        self.bot._detect_village_type = Mock(return_value='home')
        self.bot._ensure_correct_village(False)
        self.assertEqual(order, ['capture', 'sync', 'click'])
        self.bot.input.click.assert_called_once_with(1012, 1058, pause=0.15)

    def test_missing_or_empty_capture_sends_no_input(self):
        for frame in (None, np.zeros((0, 0, 3), dtype=np.uint8)):
            with self.subTest(frame=type(frame)):
                self.bot.window.screenshot.return_value = frame
                with self.assertRaisesRegex(RuntimeError, 'capture unavailable'):
                    self.bot._ensure_correct_village(False)
                self.bot.input.click.assert_not_called()

    def test_retries_stop_at_first_known_village(self):
        self.bot._detect_village_type = Mock(side_effect=[None, 'home'])
        self.bot._ensure_correct_village(False)
        self.assertEqual(self.bot._detect_village_type.call_count, 2)
        self.bot.stop_event.wait.assert_has_calls([call(0.2), call(0.3)])
        self.bot.vision.log_template_diagnostics.assert_not_called()

    def test_three_failed_attempts_log_each_portrait_once(self):
        self.bot.vision.find_template.return_value = (None, None)
        self.bot._ensure_correct_village(False)
        # Three portraits on each of the three attempts, threshold still defaults to 0.8.
        self.assertEqual(self.bot.vision.find_template.call_count, 9)
        calls = self.bot.vision.log_template_diagnostics.call_args_list
        self.assertEqual([c.args[1] for c in calls],
                         ['mbuilder.png', 'builder.png', 'gbuilder.png'])
        self.assertTrue(all(c.kwargs['region'] == (0, 0, 1920, 540) for c in calls))

    def test_stop_before_start_sends_no_input(self):
        self.bot.stop_event.is_set.return_value = True
        with self.assertRaises(InterruptedError):
            self.bot._ensure_correct_village(False)
        self.bot.window.screenshot.assert_not_called()
        self.bot.input.click.assert_not_called()

    def test_stop_during_retry_does_not_detect_again(self):
        self.bot._detect_village_type = Mock(return_value=None)
        self.bot.stop_event.wait.side_effect = [False, True]
        self.bot._ensure_correct_village(False)
        self.bot._detect_village_type.assert_called_once()
        self.bot.vision.log_template_diagnostics.assert_not_called()


class AttackTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.bot = Bot.__new__(Bot)
        self.bot.stop_event, self.bot.window, self.bot.input = Mock(), Mock(), Mock()
        self.bot.stop_event.is_set.return_value = False
        self.bot.stop_event.wait.return_value = False
        self.bot.vision = Mock()
        self.bot.vision.find_template.return_value = (None, None)
        self.bot._update_config_size = Mock()
        self.bot._dismiss_okay_or_exit_on_frame = Mock(return_value=False)
        self.bot._nudge_view_to_reveal_attack = Mock()
        self.bot._battle_rewards = Mock()
        self.bot._battle_rewards.handle.return_value = False
        self.frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(2)]
        self.bot.window.screenshot.side_effect = self.frames

    def test_timeout_keeps_best_score_and_emits_one_diagnostic(self):
        for method in ('_wait_for_attack_with_nudge', '_wait_for_image'):
            with self.subTest(method=method):
                self.bot.window.screenshot.side_effect = self.frames
                self.bot.vision.find_template_with_confidence.side_effect = [
                    (None, None, .7), (None, None, .4)]
                self.bot.vision.log_template_diagnostics.reset_mock()
                args = ('attack.png',) if method == '_wait_for_image' else ()
                with patch('app.core.bot.time.time', side_effect=[0, 0, 1, 11]):
                    self.assertEqual(getattr(self.bot, method)(*args), (None, None))
                self.bot.vision.log_template_diagnostics.assert_called_once()
                diag = self.bot.vision.log_template_diagnostics.call_args
                self.assertIs(diag.args[0], self.frames[0])
                self.assertEqual(diag.kwargs, {'region': (0, 540, 1920, 540), 'best_score': .7})
                for match in self.bot.vision.find_template_with_confidence.call_args_list:
                    self.assertEqual(match.kwargs['threshold'], .8)

    def test_success_has_no_diagnostic(self):
        self.bot.vision.find_template_with_confidence.return_value = (90, 900, .81)
        self.assertEqual(self.bot._wait_for_attack_with_nudge(), (90, 900))
        self.bot.vision.log_template_diagnostics.assert_not_called()


class VisionDiagnosticTests(unittest.TestCase):
    def setUp(self):
        cfg = SimpleNamespace(aspect_key='16_9', ref_width=2560, ref_height=1440)
        patcher = patch('app.services.vision.Config', return_value=cfg)
        patcher.start()
        self.addCleanup(patcher.stop)
        path_patcher = patch('app.services.vision.get_template_path',
                             side_effect=lambda name: Path('templates/16_9') / name)
        path_patcher.start()
        self.addCleanup(path_patcher.stop)
        self.roi = (0, 540, 1920, 540)
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.template = np.random.default_rng(8).integers(0, 256, (84, 152, 3), dtype=np.uint8)

    def test_metadata_scaling_threshold_and_roi_offsets(self):
        scaled = cv2.resize(self.template, (114, 63), interpolation=cv2.INTER_AREA)
        self.frame[700:763, 100:214] = scaled
        info = {}
        with patch('app.services.vision.cv2.imread', return_value=self.template):
            x, y, score = VisionService.find_template_with_confidence(
                self.frame, 'builder.png', threshold=.8, region=self.roi, diagnostics=info)
        self.assertEqual((x, y), (157, 731))
        self.assertGreaterEqual(score, .8)
        self.assertEqual(info['original'], (152, 84))
        self.assertEqual(info['scaled'], (114, 63))
        self.assertEqual(info['capture'], (1920, 1080))
        self.assertEqual(info['region'], self.roi)
        self.assertEqual(info['profile'], '16_9')

    def test_native_warning_is_diagnostic_only(self):
        self.frame[700:784, 100:252] = self.template
        with patch('app.services.vision.cv2.imread', return_value=self.template), \
                self.assertLogs('VisionService', level='WARNING') as logs:
            VisionService.log_template_diagnostics(self.frame, 'builder.png', self.roi)
        text = '\n'.join(logs.output)
        for field in ('best_score=', 'original=(152, 84)', 'scaled=(114, 63)',
                      'capture=(1920, 1080)', 'region=(0, 540, 1920, 540)',
                      'profile=16_9', 'scaled_score=', 'unscaled_score=1.0000',
                      'mode=reference', 'Possible native-scale template builder.png'):
            self.assertIn(field, text)

    def test_reference_template_does_not_trigger_native_warning(self):
        scaled = cv2.resize(self.template, (114, 63), interpolation=cv2.INTER_AREA)
        self.frame[700:763, 100:214] = scaled
        with patch('app.services.vision.cv2.imread', return_value=self.template), \
                self.assertLogs('VisionService', level='WARNING') as logs:
            VisionService.log_template_diagnostics(self.frame, 'builder.png', self.roi)
        self.assertNotIn('Possible native-scale', '\n'.join(logs.output))

    def test_only_attack_is_native(self):
        self.assertEqual(NATIVE_SCALE_TEMPLATES, {'attack.png'})

    def test_all_matching_paths_respect_explicit_scale_policy(self):
        # Verify every template-matching entry point, without Win32 or real PNGs.
        methods = (
            ('find_template', {'threshold': .8}),
            ('find_template_with_confidence', {'threshold': .8}),
            ('_template_best_match', {}),
            ('_match_template_top_left', {'threshold': .8}),
            ('_find_template_matches', {'threshold': .8, 'max_matches': 1}),
        )
        for name in ('attack.png', 'builder.png', 'gbuilder.png', 'mbuilder.png', 'farmbattle.png'):
            native = name == 'attack.png'
            tile = self.template if native else cv2.resize(
                self.template, (114, 63), interpolation=cv2.INTER_AREA)
            frame = np.zeros_like(self.frame)
            h, w = tile.shape[:2]
            frame[700:700+h, 100:100+w] = tile
            for method, kwargs in methods:
                with self.subTest(name=name, method=method), \
                        patch('app.services.vision.cv2.imread', return_value=self.template), \
                        patch('app.services.vision.cv2.resize', wraps=cv2.resize) as resize:
                    result = getattr(VisionService, method)(frame, name, region=self.roi, **kwargs)
                    if native:
                        resize.assert_not_called()
                    else:
                        resize.assert_called_once()
                        self.assertEqual(resize.call_args.args[1], (114, 63))
                    center = (100 + w//2, 700 + h//2)
                    if method in ('find_template', 'find_template_with_confidence'):
                        self.assertEqual(result[:2], center)
                    elif method == '_template_best_match':
                        self.assertGreaterEqual(result[0], .8)
                        self.assertEqual(result[1:], center)
                    elif method == '_match_template_top_left':
                        self.assertEqual(result[:4], (100, 700, w, h))
                        self.assertGreaterEqual(result[4], .8)
                    else:
                        self.assertEqual(result[0][:4], (100, 700, w, h))
                        self.assertGreaterEqual(result[0][4], .8)

    def test_native_metadata_and_no_automatic_scale_selection(self):
        self.frame[700:784, 100:252] = self.template
        info = {}
        with patch('app.services.vision.cv2.imread', return_value=self.template):
            x, y, score = VisionService.find_template_with_confidence(
                self.frame, 'attack.png', threshold=.8, region=self.roi, diagnostics=info)
        self.assertEqual((x, y), (176, 742))
        self.assertGreaterEqual(score, .8)
        self.assertEqual(info['original'], (152, 84))
        self.assertEqual(info['scaled'], (152, 84))

        # A native template is NOT silently retried at reference scale when it fails.
        scaled = cv2.resize(self.template, (114, 63), interpolation=cv2.INTER_AREA)
        frame = np.zeros_like(self.frame)
        frame[700:763, 100:214] = scaled
        with patch('app.services.vision.cv2.imread', return_value=self.template):
            x, y, score = VisionService.find_template_with_confidence(
                frame, 'attack.png', threshold=.8, region=self.roi)
        self.assertEqual((x, y), (None, None))
        self.assertLess(score, .8)

    def test_builder_diagnostic_always_includes_both_low_scores(self):
        with patch('app.services.vision.cv2.imread', return_value=self.template), \
                self.assertLogs('VisionService', level='WARNING') as logs:
            VisionService.log_template_diagnostics(self.frame, 'builder.png', self.roi)
        self.assertEqual(len(logs.output), 1)
        self.assertIn('scaled_score=0.0000 unscaled_score=0.0000', logs.output[0])

    def test_native_diagnostic_reports_reference_comparison_and_actual_used_size(self):
        self.frame[700:784, 100:252] = self.template
        with patch('app.services.vision.cv2.imread', return_value=self.template), \
                self.assertLogs('VisionService', level='WARNING') as logs:
            VisionService.log_template_diagnostics(self.frame, 'attack.png', self.roi)
        self.assertEqual(len(logs.output), 1)
        self.assertIn('unscaled_score=1.0000', logs.output[0])
        self.assertIn('scaled=(114, 63) used=(152, 84) mode=native', logs.output[0])

    def test_missing_template_and_capture_are_reported(self):
        with patch('app.services.vision.cv2.imread', return_value=None), \
                self.assertLogs('VisionService', level='WARNING') as logs:
            VisionService.log_template_diagnostics(self.frame, 'attack.png', self.roi)
            VisionService.log_template_diagnostics(None, 'attack.png')
        self.assertIn('template unreadable', '\n'.join(logs.output))
        self.assertIn('capture unavailable', '\n'.join(logs.output))


if __name__ == '__main__':
    unittest.main()
