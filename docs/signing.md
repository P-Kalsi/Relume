# Signing Relume.app for local development

Relume.app is code-signed after every build (`cmake/CodeSign.cmake`). Signing
seals `Info.plist`, including the camera usage string, into the signature that
macOS privacy controls (TCC) check. This is **local development signing only**:
release signing and notarization are out of scope, and Relume makes no claim
to be a signed or notarized distributable.

## Default: ad-hoc

`RELUME_CODESIGN_IDENTITY` defaults to `-` (ad-hoc). An ad-hoc designated
requirement is the code hash (`cdhash`), so it changes on every rebuild and
macOS asks for camera permission again each time. Nothing else is needed.

## Optional: a stable local identity

With a certificate, the designated requirement becomes the bundle identifier
(`dev.relume.app`) plus the certificate. Both stay the same across rebuilds,
so the camera grant persists. You create the certificate yourself; the build
never creates certificates or changes your keychain.

1. Open **Keychain Access** and choose **Keychain Access > Certificate
   Assistant > Create a Certificate…**
2. Set **Name** to `Relume Dev`, **Identity Type** to *Self-Signed Root*, and
   **Certificate Type** to *Code Signing*. Then choose **Create**.
3. Check that it is listed (it may show `CSSMERR_TP_NOT_TRUSTED`, which is
   fine for local signing):
   ```sh
   security find-identity -p codesigning
   ```
4. Configure with the identity's name or its SHA-1 hash:
   ```sh
   cmake --preset debug -DRELUME_CODESIGN_IDENTITY="Relume Dev"
   cmake --build --preset debug
   ```
   Configuration fails with instructions if the identity is not found.
   `-DRELUME_CODESIGN_IDENTITY=-` returns to ad-hoc signing.

The certificate and its private key stay in your login keychain. Never export
them into the repository. The first `codesign` run may ask for keychain
access; choose *Always Allow* only if you are comfortable with that.

Signing uses `--timestamp=none`, so it never contacts a timestamp server.

## Verifying the signature

```sh
APP=build/debug/src/Relume.app
codesign --verify --strict --verbose=2 "$APP"   # valid on disk; satisfies its Designated Requirement
codesign -dv --verbose=4 "$APP"                 # Identifier=dev.relume.app, Signature=adhoc or Authority=Relume Dev
codesign -d -r- "$APP"                          # the designated requirement
```

The designated requirement has one of two forms:
- **Ad-hoc:** `designated => cdhash H"…"`. This changes with every build.
- **Identity:** `designated => identifier "dev.relume.app" and certificate leaf = H"…"`. This stays the same across rebuilds.

## Camera permission

Launch the bundle with `open` (see `docs/build.md`) so the permission request
is attributed to Relume. Running the inner executable from a terminal
attributes it to the terminal. Relume never bypasses the macOS prompt.

To check the current grant, see **System Settings > Privacy & Security >
Camera**. `tccutil reset Camera dev.relume.app` forgets Relume's grant so the
prompt appears again. It changes your privacy settings, so run it only when you
intend to.

Status on the development Mac (2026-10-24): ad-hoc signing verified. The
identity path's configure-time check was verified to reject a missing
identity. Signing with a real local certificate has **not** been verified,
because no certificate exists on this Mac.
