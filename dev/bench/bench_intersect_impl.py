

def simple_benchmark():
    from delayed_image.channel_spec import FusedChannelSpec
    import ubelt as ub
    self = FusedChannelSpec.coerce('red|green|blue').normalize()
    other = FusedChannelSpec.coerce('red|green|blue').normalize()

    self = FusedChannelSpec.coerce(100).normalize()
    other = FusedChannelSpec.coerce(10).normalize()

    def intersection_v0(self, other):
        try:
            other_norm = ub.oset(other.normalize().parsed)
        except Exception:
            other_norm = other
        self_norm = ub.oset(self.normalize().parsed)
        new_parsed = list(self_norm & other_norm)
        new = self.__class__(new_parsed, _is_normalized=True)
        return new

    def intersection_v1(self, other):
        try:
            _other_norm = other.normalize()
            _other_parsed = _other_norm.parsed
        except Exception:
            _other_parsed = other
        _self_norm = self.normalize()
        _self_parsed = _self_norm.parsed
        _other_parsed_set = set(_other_parsed)
        new_parsed = [c for c in _self_parsed if c in _other_parsed_set]
        # Note: the previous oset implementation is concise, but too slow.
        # new_parsed = list(ub.oset(_self_parsed) & ub.oset(_other_parsed))
        new = self.__class__(new_parsed, _is_normalized=True)
        return new

    def intersection_v2(self, other):
        try:
            _other_norm = other.normalize()
            _other_parsed = _other_norm.parsed
        except Exception:
            _other_parsed = other
        _self_norm = self.normalize()
        _self_parsed = _self_norm.parsed
        _other_parsed_set = _other_parsed
        new_parsed = [c for c in _self_parsed if c in _other_parsed_set]
        new = self.__class__(new_parsed, _is_normalized=True)
        return new

    import timerit
    ti = timerit.Timerit(100, bestof=10, verbose=3)
    for timer in ti.reset('intersection_v0'):
        with timer:
            intersection_v0(self, other)

    for timer in ti.reset('intersection_v1'):
        with timer:
            intersection_v1(self, other)

    for timer in ti.reset('intersection_v2'):
        with timer:
            intersection_v2(self, other)

    print(ub.urepr(ti.rankings['mean']))
