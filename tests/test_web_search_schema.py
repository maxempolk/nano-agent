from unittest import TestCase

from pydantic import BaseModel

from core.tools.web_search import _afm_generation_schema


class _Fact(BaseModel):
    claim: str
    published_at: str = ""


class _Extraction(BaseModel):
    facts: list[_Fact] = []


class StrictSchemaTests(TestCase):
    def test_every_object_lists_all_properties_in_required(self):
        schema = _afm_generation_schema(_Extraction)

        self.assertEqual(schema["required"], ["facts"])
        item = schema["properties"]["facts"]["items"]
        self.assertEqual(sorted(item["required"]), ["claim", "published_at"])
        self.assertFalse(item.get("additionalProperties"))


if __name__ == "__main__":
    import unittest

    unittest.main()
