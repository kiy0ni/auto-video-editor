import unittest
from vortex_leads.qualifier import LeadQualifier

class TestLeadQualifier(unittest.TestCase):
    def setUp(self):
        self.q = LeadQualifier()

    def test_qualify_online(self):
        result = self.q.qualify({"type": "online"})
        self.assertEqual(result["level"], 1)
        self.assertEqual(result["price"], 0)

    def test_qualify_presencial_escritorio(self):
        r = self.q.qualify({"type": "presencial-escritorio"})
        self.assertEqual(r["level"], 2)
        self.assertEqual(r["price"], 10000)

    def test_qualify_presencial_local(self):
        r = self.q.qualify({"type": "presencial-local"})
        self.assertEqual(r["level"], 3)
        self.assertEqual(r["price"], 20000)

    def test_extract_lead_from_form(self):
        lead = self.q.extract_lead_from_form({
            "nome": "Augusto",
            "email": "augusto@vortex.ecossistema",
            "telefone": "+244 900 000 000",
            "tipoConsulta": "presencial-escritorio",
            "areaInteresse": "Tecnologia (Web, Apps, Automação)",
            "mensagem": "Teste de qualificação"
        })
        self.assertEqual(lead["name"], "Augusto")
        self.assertEqual(lead["email"], "augusto@vortex.ecossistema")
        self.assertEqual(lead["type"], "office")
        self.assertEqual(lead["level"], 2)
        self.assertEqual(lead["price"], 10000)
        self.assertEqual(lead["phone"], "+244 900 000 000")
        self.assertIn("Tecnologia", lead["area_interests"][0])
        self.assertEqual(lead["status"], "novo")

    def test_extract_lead_from_form_online(self):
        form = {
            "nome": "João",
            "email": "joao@email.com",
            "telefone": "+244 900 000 000",
            "tipoConsulta": "online",
            "areaInteresse": "",
            "mensagem": "Teste"
        }
        lead = self.q.extract_lead_from_form(form)
        self.assertEqual(lead["type"], "online")
        self.assertEqual(lead["level"], 1)
        self.assertEqual(lead["price"], 0)

if __name__ == '__main__':
    unittest.main()