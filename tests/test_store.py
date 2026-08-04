import unittest
import os
import tempfile
import sqlite3
from vortex_leads.store import LeadStore

class TestLeadStore(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.db_fd)
        self.store = LeadStore(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_create_lead_full(self):
        lead_data = {
            "name": "Augusto",
            "email": "augusto@vortex.ecossistema",
            "phone": "+244 923 456 789",
            "type": "office",
            "area_interests": ["Tecnologia", "Marketing"],
            "message": "Quero discutir o meu projecto.",
            "level": 2,
            "price": 10000,
            "status": "novo"
        }
        lead_id = self.store.create(lead_data)
        lead = self.store.get(lead_id)
        
        self.assertEqual(lead["name"], "Augusto")
        self.assertEqual(lead["email"], "augusto@vortex.ecossistema")
        self.assertEqual(lead["phone"], "+244 923 456 789")
        self.assertEqual(lead["lead_type"], "office")
        self.assertEqual(lead["message"], "Quero discutir o meu projecto.")
        self.assertEqual(lead["level"], 2)
        self.assertEqual(lead["price"], 10000)
        self.assertIn("Tecnologia", lead["area_interests"])

    def test_get_lead(self):
        lead_id = self.store.create({"name": "X", "email": "x@x", "type": "online", "level": 1, "price": 0, "status": "novo"})
        lead = self.store.get(lead_id)
        self.assertIsNotNone(lead)
        self.assertEqual(lead["name"], "X")

    def test_list_leads(self):
        self.store.create({"name": "A", "email": "a@a", "type": "online", "level": 1, "price": 0, "status": "novo"})
        self.store.create({"name": "B", "email": "b@b", "type": "online", "level": 1, "price": 0, "status": "novo"})
        self.assertEqual(len(self.store.list()), 2)

    def test_update_lead_status(self):
        lead_id = self.store.create({"name": "T", "email": "t@t", "type": "online", "level": 1, "price": 0, "status": "novo"})
        self.store.update(lead_id, {"status": "contactado"})
        self.assertEqual(self.store.get(lead_id)["status"], "contactado")

    def test_delete_lead(self):
        lead_id = self.store.create({"name": "D", "email": "d@d", "type": "online", "level": 1, "price": 0, "status": "novo"})
        self.store.delete(lead_id)
        self.assertIsNone(self.store.get(lead_id))

if __name__ == '__main__':
    unittest.main()