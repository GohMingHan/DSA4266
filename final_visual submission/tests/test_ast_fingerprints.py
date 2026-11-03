"""Stdlib-only portability and mutation checks; runnable under each interpreter."""
import ast
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ast_fingerprints import definition_fingerprint,stable_ast_dump

ROOT=Path(__file__).resolve().parents[1]
GOLDEN=json.loads((ROOT/'tests/fixtures/equivalence.json').read_text(encoding='utf-8'))


def definition(source,name):
    return next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)


class PortableFingerprints(unittest.TestCase):
    def test_all_156_unchanged_golden_hashes_without_runtime_ast_dump(self):
        trees={}
        with patch('ast.dump',side_effect=AssertionError('Runtime debug formatting must not be used')):
            for row in GOLDEN['definitions']:
                with self.subTest(module=row['module'],definition=row['name']):
                    if row['module'] not in trees:
                        trees[row['module']]=ast.parse((ROOT/'src'/str(row['module']+'.py')).read_text(encoding='utf-8'))
                    node=next(n for n in trees[row['module']].body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==row['name'])
                    expected=row.get('artifact_adaptation',{}).get('sha256',row['sha256'])
                    self.assertEqual(definition_fingerprint(node),expected)

    def test_reported_selection_label_hash_matches_original_fixture(self):
        source=(ROOT/'src/config.py').read_text(encoding='utf-8')
        node=definition(source,'selection_label')
        self.assertEqual(definition_fingerprint(node),'05390862e9d90ae7ccdbc457104a25a735e04c148963999994f765be7637e842')
        text=stable_ast_dump(node)
        self.assertIn('posonlyargs=[]',text);self.assertIn('type_params=[]',text)

    def test_absent_empty_type_params_is_equivalent_but_nonempty_is_not(self):
        node=definition('def example(x):\n    return x\n','example')
        expected=definition_fingerprint(node)
        if hasattr(node,'type_params'):del node.type_params
        self.assertEqual(definition_fingerprint(node),expected)
        node.type_params=[ast.Name(id='NewGenericParameter',ctx=ast.Load())]
        self.assertNotEqual(definition_fingerprint(node),expected)
        node=definition('def example(x):\n    return x\n','example')
        node._fields=(*node._fields,'future_execution_flag')
        node.future_execution_flag=True
        self.assertNotEqual(definition_fingerprint(node),expected)

    def test_real_policy_and_control_flow_changes_still_fail(self):
        source=(ROOT/'src/config.py').read_text(encoding='utf-8')
        original=definition(source,'Config')
        for old,new in [('4266','4267'),('1e-4','2e-4'),('frame_roc_auc','frame_pr_auc')]:
            with self.subTest(change=old):
                changed=definition(source.replace(old,new,1),'Config')
                self.assertNotEqual(definition_fingerprint(original),definition_fingerprint(changed))
        a=definition('def example(x):\n    return x > 0\n','example')
        b=definition('def example(x):\n    return x >= 0\n','example')
        self.assertNotEqual(definition_fingerprint(a),definition_fingerprint(b))

    def test_none_literals_and_list_order_remain_significant(self):
        nodes=[definition('def example():\n    return '+value+'\n','example') for value in ('None','False','[1, 2]','[2, 1]')]
        self.assertEqual(len({definition_fingerprint(node) for node in nodes}),len(nodes))
        self.assertIn('Constant(value=None)',stable_ast_dump(nodes[0]))

    def test_import_location_exception_unchanged_and_input_not_mutated(self):
        node=definition('def example():\n    from .config import Config\n    return Config()\n','example')
        before=copy.deepcopy(node)
        relocated=definition('def example():\n    from .other_module import Config\n    return Config()\n','example')
        self.assertEqual(definition_fingerprint(node),definition_fingerprint(relocated))
        self.assertEqual(stable_ast_dump(node),stable_ast_dump(before))
        standard=definition('def example():\n    import math\n    return math.pi\n','example')
        changed=definition('def example():\n    import random\n    return random.pi\n','example')
        self.assertNotEqual(definition_fingerprint(standard),definition_fingerprint(changed))


if __name__=='__main__':unittest.main()
