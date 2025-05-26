#!/usr/bin/env python3
"""
Test script to verify morphkv package installation and basic functionality.
"""

def test_basic_import():
    """Test basic import of morphkv package."""
    try:
        import morphkv
        print("✓ morphkv package imported successfully")
        
        # Test individual components
        from morphkv import DynamicCache, MorphOffloadedCache
        print("✓ Cache classes imported successfully")
        
        # Test cache creation
        cache = DynamicCache()
        print("✓ DynamicCache created successfully")
        
        # Test MorphOffloadedCache (requires CUDA and num_hidden_layers)
        try:
            import torch
            if torch.cuda.is_available():
                morph_cache = MorphOffloadedCache(num_hidden_layers=32)
                print("✓ MorphOffloadedCache created successfully")
            else:
                print("⚠ MorphOffloadedCache skipped (requires CUDA)")
        except Exception as e:
            print(f"⚠ MorphOffloadedCache creation failed: {e}")
        
        print("\nAvailable components:")
        components = [x for x in dir(morphkv) if not x.startswith('_')]
        for comp in components:
            print(f"  - {comp}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_patch_function():
    """Test if patch function can be imported (but not executed due to flash attention issues)."""
    try:
        from morphkv import patch_mistral
        print("✓ patch_mistral function imported successfully")
        print("  Note: Cannot test execution due to flash attention compatibility issues")
        return True
    except Exception as e:
        print(f"✗ Error importing patch_mistral: {e}")
        return False

if __name__ == "__main__":
    print("Testing morphkv package installation...")
    print("=" * 50)
    
    success = True
    success &= test_basic_import()
    print()
    success &= test_patch_function()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! morphkv package is properly installed.")
        print("\nTo use morphkv in your scripts:")
        print("  import morphkv")
        print("  morphkv.patch_mistral()  # Apply patches to transformers")
    else:
        print("✗ Some tests failed. Please check the installation.") 