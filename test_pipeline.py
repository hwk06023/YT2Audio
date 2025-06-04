#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path


def check_dependencies():
    print("🔍 Checking Pipeline Dependencies...\n")

    issues = []

    print("1. Checking Python version...")
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
        print("   ❌ Python 3.8+ required")
    else:
        print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}")

    print("\n2. Checking required packages...")
    required_packages = ["pydub", "elevenlabs", "python-dotenv", "textgrids"]

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            issues.append(f"Missing package: {package}")
            print(f"   ❌ {package}")

    print("\n3. Checking FFmpeg...")
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ FFmpeg")
        else:
            issues.append("FFmpeg not working properly")
            print("   ❌ FFmpeg not working")
    except FileNotFoundError:
        issues.append("FFmpeg not installed")
        print("   ❌ FFmpeg not found")

    print("\n4. Checking MFA...")
    try:
        result = subprocess.run(["mfa", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ MFA")
        else:
            issues.append("MFA not working properly")
            print("   ❌ MFA not working")
    except FileNotFoundError:
        issues.append("MFA not installed")
        print("   ❌ MFA not found")

    print("\n5. Checking environment file...")
    if os.path.exists(".env"):
        print("   ✅ .env file exists")
        from dotenv import load_dotenv

        load_dotenv()
        if os.getenv("ELEVENLABS_API_KEY"):
            print("   ✅ ElevenLabs API key found")
        else:
            issues.append("ELEVENLABS_API_KEY not set in .env")
            print("   ❌ ELEVENLABS_API_KEY not set")
    else:
        issues.append(".env file missing")
        print("   ❌ .env file not found")

    print("\n6. Checking data directory...")
    if os.path.exists("data"):
        audio_files = [f for f in os.listdir("data") if f.endswith(".wav")]
        if audio_files:
            print(f"   ✅ Data directory with {len(audio_files)} audio files")
            for f in audio_files[:3]:
                print(f"      - {f}")
            if len(audio_files) > 3:
                print(f"      ... and {len(audio_files) - 3} more")
        else:
            issues.append("No audio files in data directory")
            print("   ❌ No audio files found")
    else:
        issues.append("Data directory missing")
        print("   ❌ Data directory not found")

    print("\n" + "=" * 50)

    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n📋 Setup instructions:")
        print("   1. Install missing packages: pip install -r requirements.txt")
        print("   2. Install FFmpeg: https://ffmpeg.org/download.html")
        print("   3. Install MFA: pip install montreal-forced-alignment")
        print("   4. Create .env file with ELEVENLABS_API_KEY")
        print("   5. Place audio files in data/ directory")
        return False
    else:
        print("✅ All dependencies satisfied! Pipeline ready to run.")
        return True


def test_basic_functionality():
    print("\n🧪 Testing Basic Functionality...\n")

    try:
        from Integrated_pipline import AudioProcessingPipeline

        print("   ✅ Pipeline import successful")

        audio_files = [
            f.replace(".wav", "") for f in os.listdir("data") if f.endswith(".wav")
        ]
        if audio_files:
            test_file = audio_files[0]
            print(f"   ✅ Test audio file: {test_file}")

            pipeline = AudioProcessingPipeline(test_file)
            print("   ✅ Pipeline initialization successful")

            return True
        else:
            print("   ❌ No audio files for testing")
            return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    print("🚀 Audio Processing Pipeline - System Check\n")

    deps_ok = check_dependencies()

    if deps_ok:
        func_ok = test_basic_functionality()

        if func_ok:
            print("\n🎉 System ready! You can now run:")
            print("   python demo_pipeline.py")
            print("   or")
            print("   python Integrated_pipline.py <audio_name>")
        else:
            print("\n⚠️  Dependencies OK but functionality test failed")
    else:
        print("\n⚠️  Please fix the issues above before proceeding")


if __name__ == "__main__":
    main()
