# -*- mode: python ; coding: utf-8 -*-

PY_FILE = 'VisionUi.py'
NAME = PY_FILE.strip('.py')
block_cipher = None

a = Analysis([PY_FILE],
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
splash = Splash('resources/SplashScreen.png',
                binaries=a.binaries,
                datas=a.datas,
                text_pos=None,
                text_size=12,
                minify_script=True)

a.datas = [fp for fp in a.datas if 'pylonCXP' not in fp[0]]  # Remove pylonCXP
a.datas = [fp for fp in a.datas if '_genicam' not in fp[0]]  # Remove _genicam.cp39-win_amd64.pyd which is already in binaries
a.datas = [fp for fp in a.datas if '_pylon' not in fp[0]]    # Remove _pylon.cp39-win_amd64.pyd which is already in binaries

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas, 
          splash, 
          splash.binaries,
          [],
          name=NAME,
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None,
          icon='resources/CameraLogo_Transparent.ico')

# Compile in terminal: pyinstaller VisionUi.spec