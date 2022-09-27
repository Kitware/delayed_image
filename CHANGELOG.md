# Changelog

We are currently working on porting this changelog to the specifications in
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Version 0.2.1 - Unreleased


## [Version 0.2.0] -

### Added
* new method `get_transform_from` which returns the transform from the space of
  one delayed image to the space of another.

### Changed
* Added `noop_eps` parameter to delayed warp, which will optimize it away if
  the warp is close to identity.

### Fixed
* Fixed issue where warp parameters would not be carried through. 


## [Version 0.1.0] -

### Added
* Initial version ported from kwcoco
