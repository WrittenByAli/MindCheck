"""Verify that all MindCheck dependencies are importable and report their versions."""
import importlib
import sys

REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("chess", "python-chess"),
    ("streamlit", "streamlit"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("yaml", "pyyaml"),
    ("pydantic", "pydantic"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("pytest", "pytest"),
    ("tqdm", "tqdm"),
]


def main() -> int:
    print(f"Python: {sys.version}\n")
    failed = []

    for module_name, package_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {package_name:<16} {version}")
        except ImportError as exc:
            print(f"[MISSING] {package_name:<16} {exc}")
            failed.append(package_name)

    print()
    try:
        import torch

        if torch.cuda.is_available():
            print(f"PyTorch backend: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            print("PyTorch backend: CPU")
    except ImportError:
        pass

    if failed:
        print(f"\n{len(failed)} package(s) missing: {', '.join(failed)}")
        return 1

    print("\nAll dependencies verified successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
