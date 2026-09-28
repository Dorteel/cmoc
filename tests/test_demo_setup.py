"""Verify demo setup and cleanup without starting Webots or memory backends."""

import unittest
from unittest.mock import Mock, patch

import demo


class DemoSetupTests(unittest.TestCase):
    def setUp(self):
        patcher = patch('demo.rclpy')
        self.ros = patcher.start()
        self.addCleanup(patcher.stop)
        self.ros.ok.return_value = True
        perception_patcher = patch('demo.PerceptionLauncher')
        self.perception_factory = perception_patcher.start()
        self.addCleanup(perception_patcher.stop)
        self.perception = self.perception_factory.return_value

    def tearDown(self):
        self.ros.init.assert_called_once_with(
            args=[], signal_handler_options=demo.SignalHandlerOptions.NO)
        self.ros.shutdown.assert_called_once_with()

    @patch('demo.RoomNavigator')
    @patch('demo.spa_loop')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py'])
    def test_setup_precedes_memory_and_ctrl_c_stops_simulator(self, factory, run, navigator):
        calls = Mock()
        calls.attach_mock(self.ros, 'ros')
        calls.attach_mock(factory.return_value, 'simulator')
        calls.attach_mock(run, 'memory')
        calls.attach_mock(self.perception, 'perception')
        calls.attach_mock(navigator.return_value, 'navigator')
        factory.return_value.is_running.side_effect = KeyboardInterrupt
        demo.main()
        navigator.assert_called_once()
        self.perception_factory.assert_called_once_with(backend="nebula")
        self.assertEqual([call[0] for call in calls.mock_calls],
                         ['ros.init', 'simulator.start', 'simulator.wait_until_ready',
                          'navigator.wait_until_ready', 'perception.start', 'perception.wait_for_observe_with_vlm', 'memory',
                          'simulator.is_running', 'navigator.close', 'perception.stop', 'simulator.stop',
                          'ros.ok', 'ros.shutdown'])

    @patch('demo.RoomNavigator')
    @patch('demo.spa_loop')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py'])
    def test_readiness_failure_stops_before_initializing_memory(self, factory, run, navigator):
        factory.return_value.wait_until_ready.side_effect = TimeoutError('not ready')
        with self.assertRaises(TimeoutError):
            demo.main()
        run.assert_not_called()
        factory.return_value.stop.assert_called_once()

    @patch('demo.RoomNavigator')
    @patch('demo.spa_loop')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py', '--no-simulator'])
    def test_no_simulator_runs_existing_demo(self, factory, run, navigator):
        demo.main()
        factory.return_value.start.assert_not_called()
        factory.return_value.wait_until_ready.assert_not_called()
        run.assert_called_once()


    @patch('demo.RoomNavigator')
    @patch('demo.run_demo')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py', '--test-navigation'])
    def test_movement_flag_passes_ready_navigator(self, factory, run, navigator):
        factory.return_value.is_running.side_effect = KeyboardInterrupt
        demo.main()
        self.perception_factory.assert_not_called()
        navigator.return_value.wait_until_ready.assert_called_once()
        run.assert_called_once_with(navigator.return_value)
        navigator.return_value.close.assert_called_once()
        factory.return_value.stop.assert_called_once()



    @patch('demo.RoomNavigator')
    @patch('demo.run_demo')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py', '--test-navigation'])
    def test_cleanup_failure_still_shuts_down_ros(self, factory, run, navigator):
        factory.return_value.is_running.side_effect = KeyboardInterrupt
        navigator.return_value.close.side_effect = RuntimeError('cleanup failed')
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            demo.main()
        factory.return_value.stop.assert_called_once()

    @patch('demo.SimulatorLauncher', side_effect=RuntimeError('construction failed'))
    @patch('sys.argv', ['demo.py'])
    def test_construction_failure_still_shuts_down_ros(self, factory):
        with self.assertRaisesRegex(RuntimeError, 'construction failed'):
            demo.main()



    @patch('demo.RoomNavigator')
    @patch('demo.run_demo')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py', '--test-navigation'])
    def test_inactive_nav2_blocks_movement_and_cleans_up(self, factory, run, navigator):
        navigator.return_value.wait_until_ready.side_effect = RuntimeError('bt_navigator=inactive')
        with self.assertRaisesRegex(RuntimeError, 'bt_navigator=inactive'):
            demo.main()
        run.assert_not_called()
        navigator.return_value.go_to_room.assert_not_called()
        navigator.return_value.close.assert_called_once()
        factory.return_value.stop.assert_called_once()



    @patch('demo.RoomNavigator')
    @patch('demo.spa_loop')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py'])
    def test_perception_timeout_prevents_spa_and_cleans_up(self, factory, run, navigator):
        self.perception.wait_for_observe_with_vlm.side_effect = RuntimeError('server not ready')
        with self.assertRaisesRegex(RuntimeError, 'server not ready'):
            demo.main()
        run.assert_not_called()
        self.perception.stop.assert_called_once()
        factory.return_value.stop.assert_called_once()

    @patch('demo.spa_loop')
    @patch('sys.argv', ['demo.py', '--no-simulator'])
    def test_normal_completion_stops_perception(self, run):
        demo.main()
        self.perception.start.assert_called_once()
        self.perception.wait_for_observe_with_vlm.assert_called_once()
        self.perception.stop.assert_called_once()



    @patch('demo.candidate_locations', return_value=[])
    @patch('demo.observe_scene_with_vlm', return_value={'objects': [], 'relations': []})
    @patch('demo.RoomNavigator')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py'])
    def test_instruction_follows_all_readiness_and_banner(self, factory, navigator, observe, candidates):
        factory.return_value.is_running.side_effect = KeyboardInterrupt
        with patch('builtins.print') as output:
            def instruction(prompt):
                factory.return_value.wait_until_ready.assert_called_once()
                navigator.return_value.wait_until_ready.assert_called_once()
                self.perception.wait_for_observe_with_vlm.assert_called_once()
                self.perception_factory.assert_called_once_with(backend='nebula')
                self.assertIn('CMOC READY\nPerception backend: Nebula', output.call_args.args[0])
                observe.assert_not_called()
                return ''
            with patch('builtins.input', side_effect=instruction) as ask:
                demo.main()
                ask.assert_called_once_with('Instruction [Bring me a fork]: ')
        observe.assert_called_once_with(schema_path='schemas/objects.json')

    @patch('builtins.input')
    @patch('demo.spa_loop')
    @patch('demo.RoomNavigator')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py'])
    def test_missing_key_setup_failure_never_prompts(self, factory, navigator, run, ask):
        self.perception.start.side_effect = RuntimeError('NEBULA_API_KEY is not set')
        with self.assertRaisesRegex(RuntimeError, 'NEBULA_API_KEY is not set'):
            demo.main()
        run.assert_not_called()
        ask.assert_not_called()
        navigator.return_value.close.assert_called_once()
        factory.return_value.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
