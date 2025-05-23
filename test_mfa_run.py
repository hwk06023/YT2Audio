import os
import subprocess

# 환경 변수 설정
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

print("Testing MFA command line...")

# 간단한 MFA 명령어 실행
result = subprocess.run(['mfa', 'version'], capture_output=True, text=True)
print(f"MFA version: {result.stdout}")
if result.stderr:
    print(f"Error: {result.stderr}")

# MFA validate 명령어 테스트 (실제 파일 없어도 실행 가능)
print("\nTesting MFA validate command...")
result = subprocess.run(['mfa', 'validate', '--help'], capture_output=True, text=True)
if "Error #15" in result.stderr:
    print("OpenMP 충돌 발생!")
else:
    print("validate 명령어 도움말 표시 성공")
