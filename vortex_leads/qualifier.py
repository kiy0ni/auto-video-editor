class LeadQualifier:
    def qualify(self, lead_data):
        # Mapear diretamente os campos do formulário Netlify para a estrutura VORTEX
        tipo = lead_data.get("type")
        
        if tipo == "online":
            return {"level": 1, "price": 0}
        elif tipo == "presencial-escritorio":
            return {"level": 2, "price": 10000}
        elif tipo == "presencial-local":
            return {"level": 3, "price": 20000}
        else:
            return {"level": 0, "price": 0}
    
    def extract_lead_from_form(self, form_data):
        # Normalizar o tipo do formulário Netlify para o campo "type"
        tipo_form = form_data.get("tipoConsulta", "")
        tipo_normalizado = {
            "online": "online",
            "presencial-escritorio": "presencial-escritorio",
            "presencial-local": "presencial-local"
        }.get(tipo_form, "online")
        
        # Qualificar usando o tipo normalizado
        qualified = self.qualify({"type": tipo_normalizado})
        
        # Mapear para a estrutura interna VORTEX
        lead_type_map = {
            "online": "online",
            "presencial-escritorio": "office",
            "presencial-local": "client_site"
        }
        
        return {
            "name": form_data.get("nome", "").strip(),
            "email": form_data.get("email", "").strip(),
            "phone": form_data.get("telefone", "").strip(),
            "type": lead_type_map.get(tipo_normalizado, "online"),
            "area_interests": [form_data.get("areaInteresse", "").strip()] if form_data.get("areaInteresse") else [],
            "message": form_data.get("mensagem", "").strip(),
            "level": qualified["level"],
            "price": qualified["price"],
            "status": "novo"
        }
