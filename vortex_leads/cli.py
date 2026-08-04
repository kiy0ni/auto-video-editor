#!/usr/bin/env python3
"""VORTEX Lead System — CLI principal."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vortex_leads.qualifier import LeadQualifier
from vortex_leads.store import LeadStore


def main():
    parser = argparse.ArgumentParser(
        description="VORTEX Lead System — Qualificação e Automação de Leads"
    )
    parser.add_argument(
        "--db", default=os.path.expanduser("~/vortex_leads.db"),
        help="Caminho para o ficheiro SQLite (padrão: ~/vortex_leads.db)"
    )
    sub = parser.add_subparsers(dest="command")

    # Qualificar lead
    q = sub.add_parser("qualify", help="Qualifica um lead segundo as regras VORTEX")
    q.add_argument("--type", required=True, choices=["online", "office", "client_site"])
    q.add_argument("--name", required=True)
    q.add_argument("--email", required=True)

    # Listar leads
    sub.add_parser("list", help="Lista todos os leads")

    # Adicionar lead manualmente
    a = sub.add_parser("add", help="Adiciona um lead manualmente")
    a.add_argument("--name", required=True)
    a.add_argument("--email", required=True)
    a.add_argument("--type", required=True, choices=["online", "office", "client_site"])

    # Actualizar status
    u = sub.add_parser("update", help="Actualiza o status de um lead")
    u.add_argument("--id", type=int, required=True)
    u.add_argument("--status", required=True)

    # Remover lead
    d = sub.add_parser("delete", help="Remove um lead")
    d.add_argument("--id", type=int, required=True)

    args = parser.parse_args()
    store = LeadStore(args.db)
    qualifier = LeadQualifier()

    if args.command == "qualify":
        result = qualifier.qualify({"type": args.type})
        lead_id = store.create({
            "name": args.name,
            "email": args.email,
            "type": args.type,
            "level": result["level"],
            "price": result["price"],
            "status": "novo",
        })
        print(f"✅ Lead qualificado e guardado (id={lead_id})")
        print(f"   Nível: {result['level']} | Preço: {result['price']} Kz")

    elif args.command == "list":
        leads = store.list()
        if not leads:
            print("Nenhum lead encontrado.")
            return
        for lead in leads:
            print(f"[{lead['id']}] {lead['name']} ({lead['email']}) — "
                  f"Nível {lead['level']} — {lead['price']} Kz — {lead['status']}")

    elif args.command == "add":
        result = qualifier.qualify({"type": args.type})
        lead_id = store.create({
            "name": args.name,
            "email": args.email,
            "type": args.type,
            "level": result["level"],
            "price": result["price"],
            "status": "novo",
        })
        print(f"✅ Lead adicionado (id={lead_id})")

    elif args.command == "update":
        store.update(args.id, {"status": args.status})
        print(f"✅ Lead {args.id} actualizado para '{args.status}'")

    elif args.command == "delete":
        store.delete(args.id)
        print(f"✅ Lead {args.id} removido")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()