# Changelog

We are currently working on porting this changelog to the specifications in
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Version 0.2.5 - Unreleased

### Changed
* Small speedups


## Version 0.2.4 - Released 2023-02-02

### Changed

* modified the signature of `DelayedLoad.demo` to better match `__init__` and
  `grab_image_test_fpath`.

### Fixed

* floating point bug where an overview would fail to be absorbed into a warp.


## Version 0.2.3 - Released 2022-11-07

### Fixed
* Issue in SensorChan spec where duplicate sensors were present in concise codes

## Version 0.2.2 - Released 2022-09-28

### Added
* Added `__json__` to sensorchan specs
* Added `resize` method

### Fixed
* A Dequantize node between a warp and and overview now has its size modified correctly.

### Changed
* `write_network_text` now defaults to rich='auto'

### Changed
* 3.6 support

## [Version 0.2.0] - Released 2022-09-27

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
