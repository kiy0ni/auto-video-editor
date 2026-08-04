#!/usr/bin/env python3
"""Validação completa do VORTEX Lead System."""

import subprocess
import sys
import os

def run_test_file(file_path):
    """Executa um ficheiro de teste com unittest."""
    cmd = [
        "python3",
        "-m", "unittest",
        file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def main():
    print("🔍 VALIDANDO VORTEX LEAD SYSTEM")
    print("=" * 50)
    
    test_files = [
        "/home/augusto/vortex-pro-gh/tests/test_qualifier.py",
        "/home/augusto/vortex-pro-gh/tests/test_store.py"
    ]
    
    all_passed = True
    for test_file in test_files:
        passed, stdout, stderr = run_test_file(test_file)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{status} {os.path.basename(test_file)}")
        if not passed:
            print("STDOUT:")
            print(stdout)
            print("STDERR:")
            print(stderr)
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 TODOS OS TESTES PASSARAM! O VORTEX LEAD SYSTEM ESTÁ PRONTO.")
        print("🚀 CLI está disponível para uso imediato.")
    else:
        print("💥 ALGUNS TESTES FALHARAM. REVISAR O CÓDIGO.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)