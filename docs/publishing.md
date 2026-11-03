# Publishing to Obsidian Community Plugins

This is a prepared development release, not an existing marketplace listing. The selected repository is [toolanzyhhh1234/heading-search](https://github.com/toolanzyhhh1234/heading-search), with Ian Forester Chou as publisher. Directory submission requires the owner's Obsidian account linked to that GitHub account.

The current official process is documented in [Submit your plugin](https://docs.obsidian.md/Plugins/Releasing/Submit%20your%20plugin). It uses the [Obsidian Community directory](https://community.obsidian.md/), rather than assuming the older community-plugins.json pull-request process.

1. Set the manifest author to the intended publisher. Confirm that `heading-search` is available as a unique directory ID. Check `minAppVersion` on the oldest Obsidian version you intend to support; this preview uses APIs available by 1.8.0 but has not been tested on every older desktop build.
2. Publish the source, README, LICENSE, manifest, dependency lockfile, and notices in a public GitHub repository. Do not commit downloaded datasets, model weights, caches, or test profiles.
3. Run `npm ci`, `npm test`, and `npm run package`. Test the release in an actual desktop Obsidian vault, including model setup, two-vault reuse, offline use, and reload/unload.
4. Create a Git tag matching the manifest exactly: `0.1.0`, without a `v` prefix. The included workflow builds and creates a **draft** GitHub release for review.
5. Attach `main.js`, `manifest.json`, and `styles.css` individually to the release. The ZIP is for manual installation; it does not replace these three assets. Publish the reviewed release.
6. Sign in to the Community directory with the owner's Obsidian account, link GitHub, add the plugin, and resolve the directory's scanner/review findings. The manifest on the default branch must match the intended release.

The release packages the JavaScript worker and ONNX WebAssembly runtime inside `main.js`. Community installation therefore needs only the standard release assets. The model is an explicit optional data download; there is no dynamic dependency installer or remote JavaScript loader.

The README discloses both network use and access to the shared cache outside the vault, as required by [Developer policies](https://docs.obsidian.md/community-directory/developer-policies). Desktop-only status is required because the cache uses Node filesystem APIs; see [submission requirements](https://docs.obsidian.md/community-directory/submission-requirements-for-plugins).

Marketplace acceptance and ownership verification are external steps. A successful local build does not imply directory approval.
