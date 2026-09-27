"""Real RoboKGNet integration; writes are confined to a temporary data copy."""

from pathlib import Path
import shutil
import tempfile
import unittest

from knowledge_interface import KnowledgeInterface


DATA = (Path(__file__).resolve().parents[1] / 'external' / 'robokgnet'
        / 'robonet_graph')


class KnowledgeInterfaceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge = KnowledgeInterface()

    def test_real_concepts(self):
        k = self.knowledge
        self.assertIn(k.get_concept('car.n.01'), k.resolve_concept('car'))
        self.assertIn('automobile', k.get_alternative_names('car.n.01'))
        self.assertEqual(k.resolve_concept('automobile'), [k.get_concept('car.n.01')])
        self.assertEqual(k.get_superclasses('car.n.01'), ['motor_vehicle.n.01'])
        self.assertEqual(k.resolve_concept('aircraft'), [k.get_concept('aircraft.n.01')])
        self.assertEqual(k.get_locations('aircraft.n.01'), [('sky.n.01', 1)])

    def test_real_action_and_framenet_elements(self):
        k = self.knowledge
        action = k.get_action('bring-11.3')
        self.assertEqual(k.resolve_action('bring-11.3'), [action])
        self.assertEqual(action['framenet_frame'], 'Bringing')
        self.assertEqual(k.get_action_frame('bring-11.3'), action['frame'])
        self.assertEqual(k.get_frame_roles('bring-11.3'),
                         ['agent', 'destination', 'source', 'theme'])
        # This revision has no enriched role mappings. Preserve upstream nulls;
        # the FrameNet Theme description is stored among additional elements.
        for role in k.get_frame_roles('bring-11.3'):
            self.assertIsNone(k.get_role('bring-11.3', role))
        elements = k.get_additional_frame_elements('bring-11.3')
        self.assertEqual(elements, action['additional_frame_elements'])
        self.assertIn('The objects being carried.', elements['Theme']['description'])
        self.assertIn('beginning of the path', elements['Source']['description'])
        self.assertIn('Carrier', elements)

    def test_real_evoking_word(self):
        action = self.knowledge.get_action('bring-11.3')
        self.assertIn('carry', action['evoking_words'])
        self.assertNotIn('carry', action['members'])
        self.assertEqual(self.knowledge.resolve_action('carry'), [action])

    def test_location_update_save_reload_on_temporary_copy(self):
        # Copy bytes only: CMOC never parses the JSON. All queries and writes
        # below pass through the facade into RoboKGNet.
        with tempfile.TemporaryDirectory() as directory:
            concepts = Path(directory) / 'concepts.json'
            shutil.copyfile(DATA / 'robokgconceptnet.json', concepts)
            k = KnowledgeInterface(concepts_path=concepts)
            before = dict(k.get_locations('aircraft.n.01'))['sky.n.01']
            k.update_location('aircraft.n.01', 'sky.n.01')
            self.assertEqual(dict(k.get_locations('aircraft.n.01'))['sky.n.01'], before + 1)
            self.assertEqual(dict(KnowledgeInterface(concepts).get_locations(
                'aircraft.n.01'))['sky.n.01'], before)
            k.save()
            reloaded = KnowledgeInterface(concepts)
            self.assertEqual(dict(reloaded.get_locations('aircraft.n.01'))['sky.n.01'],
                             before + 1)
            reloaded.update_location('aircraft.n.01', 'sky.n.01', increment=2)
            self.assertEqual(dict(reloaded.get_locations('aircraft.n.01'))['sky.n.01'],
                             before + 3)
        self.assertEqual(KnowledgeInterface().get_locations('aircraft.n.01'),
                         [('sky.n.01', 1)])


if __name__ == '__main__':
    unittest.main()
