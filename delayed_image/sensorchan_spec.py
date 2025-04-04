"""
This is an extension of :mod:`delayed_image.channel_spec`, which augments channel
information with an associated sensor attribute. Eventually, this will entirely
replace the channel spec.

Example:
    >>> # xdoctest: +REQUIRES(module:lark)
    >>> # hack for 3.6
    >>> from delayed_image import sensorchan_spec
    >>> import delayed_image
    >>> delayed_image.SensorChanSpec = sensorchan_spec.SensorChanSpec
    >>> self = delayed_image.SensorChanSpec.coerce('sensor0:B1|B8|B8a|B10|B11,sensor1:B11|X.2|Y:2:6,sensor2:r|g|b|disparity|gauss|B8|B11,sensor3:r|g|b|flowx|flowy|distri|B10|B11')
    >>> self.normalize()
"""

import ubelt as ub
import itertools as it
import functools


try:
    cache = functools.cache
except AttributeError:
    cache = ub.memoize

try:
    from lark import Transformer
except ImportError:
    class FakeTransformer:
        pass
    # TODO: get xdev typetubs to ignore this
    # probably need some kind of directive.
    Transformer = FakeTransformer

SENSOR_CHAN_GRAMMAR = ub.codeblock(
    '''
    // SENSOR_CHAN_GRAMMAR
    ?start: stream

    // An identifier can contain spaces
    IDEN: ("_"|"*"|LETTER) ("_"|" "|"-"|"*"|LETTER|DIGIT)*

    chan_single : IDEN
    chan_getitem : IDEN "." INT
    chan_getslice_0b : IDEN ":" INT
    chan_getslice_ab : (IDEN "." INT ":" INT) | (IDEN ":" INT ":" INT)

    // A channel code can just be an ID, or it can have a getitem
    // style syntax with a scalar or slice as an argument
    chan_code : chan_single | chan_getslice_0b | chan_getslice_ab | chan_getitem

    // Fused channels are an ordered sequence of channel codes (without sensors)
    fused : chan_code ("|" chan_code)*

    // Channels can be specified in a sequence but must contain parens
    fused_seq : "(" fused ("," fused)* ")"

    // Sensors can be specified in a sequence but must contain parens
    sensor_seq : "(" IDEN ("," IDEN)* "):"

    sensor_lhs : (IDEN ":") | (sensor_seq)

    // A channel only part can be a fused channel or a sequence
    channel_rhs : fused | fused_seq

    sensor_chan : sensor_lhs channel_rhs?

    nosensor_chan : channel_rhs

    stream_item : sensor_chan | nosensor_chan

    // A stream is an unordered sequence of fused channels, that can
    // optionally contain sensor specifications.

    stream : stream_item ("," stream_item)*

    %import common.DIGIT
    %import common.LETTER
    %import common.INT
    ''')


"""
TODO: add the concept of an exclusive or operator with left hand priority. The
idea is that we can specify a code that will use the one fused channel spec if
it is available, but if it is not, it will fall back to the next one. E.G.


WV:((red|green|blue)^(pan))

(L8,S2,WV,WV1):((red|green|blue)^(pan))


Possible Production Rules:

    fused : chan_code ("|" chan_code)*
    fused_seq : "(" fused ("," fused)* ")"



Maybe also include that on the sensor side?

Use WV:r|g|b if we have it otherwise use S2:r|g|b

(WV^S2)(r|g|b)

"""


class SensorSpec(ub.NiceRepr):
    """
    A simple wrapper for sensors in case we want to do anything fancy with them
    later. For now they are just a string.
    """
    def __init__(self, spec):
        self.spec = spec

    def __nice__(self):
        return self.spec

    def __json__(self):
        return self.spec


class SensorChanSpec(ub.NiceRepr):
    """
    The public facing API for the sensor / channel specification

    Note:
        Using the explicit constructor should be avoided, as it will likely
        change for implementation efficiency in the future. Use `coerce` and
        `from_spec` instead. (Or submit a PR/MR with additional constructors
        you might find useful)

    Example:
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> from delayed_image.sensorchan_spec import SensorChanSpec
        >>> self = SensorChanSpec.from_spec('(L8,S2):BGR,WV:BGR,S2:nir,L8:land.0:4')
        >>> s1 = self.normalize()
        >>> s2 = self.concise()
        >>> streams = self.streams()
        >>> print(s1)
        >>> print(s2)
        >>> print('streams = {}'.format(ub.urepr(streams, sv=1, nl=1)))
        L8:BGR,S2:BGR,WV:BGR,S2:nir,L8:land.0|land.1|land.2|land.3
        (L8,S2,WV):BGR,L8:land:4,S2:nir
        streams = [
            L8:BGR,
            S2:BGR,
            WV:BGR,
            S2:nir,
            L8:land.0|land.1|land.2|land.3,
        ]

    Example:
        >>> # Check with generic sensors
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> from delayed_image.sensorchan_spec import SensorChanSpec
        >>> import delayed_image
        >>> self = SensorChanSpec.from_spec('(*):BGR,*:BGR,*:nir,*:land.0:4')
        >>> self.concise().normalize()
        >>> s1 = self.normalize()
        >>> s2 = self.concise()
        >>> print(s1)
        >>> print(s2)
        *:BGR,*:BGR,*:nir,*:land.0|land.1|land.2|land.3
        (*):BGR,*:(nir,land:4)
        >>> import delayed_image
        >>> c = delayed_image.ChannelSpec.coerce('BGR,BGR,nir,land.0:8')
        >>> c1 = c.normalize()
        >>> c2 = c.concise()
        >>> print(c1)
        >>> print(c2)

    Example:
        >>> # Check empty channels
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> from delayed_image.sensorchan_spec import SensorChanSpec
        >>> import delayed_image
        >>> print(SensorChanSpec.from_spec('*:').normalize())
        *:
        >>> print(SensorChanSpec.from_spec('sen:').normalize())
        sen:
        >>> print(SensorChanSpec.from_spec('sen:').normalize().concise())
        sen:
        >>> print(SensorChanSpec.from_spec('sen:').concise().normalize().concise())
        sen:
    """
    def __init__(self, spec: str):
        self.spec: str = spec

    def __nice__(self):
        return self.spec

    def __json__(self):
        return self.spec

    def __str__(self):
        return self.spec

    def numel(self) -> int:
        return sum(s.numel() for s in self.streams())

    @classmethod
    def from_spec(cls, spec: str) -> 'SensorChanSpec':
        """
        Explicit constructor from a string specification.
        """
        self = cls(spec)
        return self

    @classmethod
    def coerce(cls, data) -> 'SensorChanSpec':
        """
        Attempt to interpret the data as a channel specification

        Returns:
            SensorChanSpec

        Example:
            >>> # xdoctest: +REQUIRES(module:lark)
            >>> from delayed_image.sensorchan_spec import *  # NOQA
            >>> from delayed_image.sensorchan_spec import SensorChanSpec
            >>> data = SensorChanSpec.coerce(3)
            >>> assert SensorChanSpec.coerce(data).normalize().spec == '*:u0|u1|u2'
            >>> data = SensorChanSpec.coerce(3)
            >>> assert data.spec == 'u0|u1|u2'
            >>> assert SensorChanSpec.coerce(data).spec == 'u0|u1|u2'
            >>> data = SensorChanSpec.coerce('u:3')
            >>> assert data.normalize().spec == '*:u.0|u.1|u.2'
        """
        import delayed_image
        if isinstance(data, cls):
            self = data
            return self
        elif isinstance(data, str):
            self = cls.from_spec(data)
            return self
        elif isinstance(data, delayed_image.FusedChannelSpec):
            spec = data.spec
            self = cls.from_spec(spec)
            return self
        elif isinstance(data, delayed_image.ChannelSpec):
            spec = data.spec
            self = cls.from_spec(spec)
            return self
        else:
            chan = delayed_image.ChannelSpec.coerce(data)
            self = cls.from_spec(chan.spec)
            return self

    def normalize(self) -> 'SensorChanSpec':
        new_spec = normalize_sensor_chan(self.spec)
        new = self.__class__.from_spec(new_spec)
        return new

    def concise(self) -> 'SensorChanSpec':
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:lark)
            >>> from delayed_image import SensorChanSpec
            >>> a = SensorChanSpec.coerce('Cam1:(red,blue)')
            >>> b = SensorChanSpec.coerce('Cam2:(blue,green)')
            >>> c = (a + b).concise()
            >>> print(c)
            (Cam1,Cam2):blue,Cam1:red,Cam2:green
            >>> # Note the importance of parenthesis in the previous example
            >>> # otherwise channels will be assigned to `*` the generic sensor.
            >>> a = SensorChanSpec.coerce('Cam1:red,blue')
            >>> b = SensorChanSpec.coerce('Cam2:blue,green')
            >>> c = (a + b).concise()
            >>> print(c)
            (*,Cam2):blue,*:green,Cam1:red
        """
        new_spec = concise_sensor_chan(self.spec)
        new = self.__class__.from_spec(new_spec)
        return new

    def streams(self) -> list:
        """
        Returns:
            List[FusedSensorChanSpec]:
                List of sensor-names and fused channel specs
        """
        parts = sensorchan_normalized_parts(self.spec)
        streams = [
            FusedSensorChanSpec(SensorSpec(part.sensor), part.chan.data)
            for part in parts]
        return streams

    def split(self) -> 'SensorChanSpec':
        """
        Split each channel into a separate stream

        Returns:
            SensorChanSpec

        Example:
            >>> # xdoctest: +REQUIRES(module:lark)
            >>> from delayed_image import SensorChanSpec
            >>> self = SensorChanSpec.coerce('Cam1:(red,blue),Cam1:feat.0:3')
            >>> print(self.split().concise())
            Cam1:(red,blue,feat.0,feat.1,feat.2)
        """
        # TODO: would be better to work with a fast and consistent
        # backend data structure rather than doing everything with strings.
        new_subspecs = []
        for stream in self.streams():
            sensor_spec = stream.sensor.spec
            for subchan in stream.chans.split().streams():
                new_subspec = f'{sensor_spec}:{subchan.spec}'
                new_subspecs.append(new_subspec)
        new_spec = ','.join(new_subspecs)
        separted = SensorChanSpec.from_spec(new_spec)
        return separted

    def late_fuse(self, *others):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:lark)
            >>> import delayed_image
            >>> from delayed_image import sensorchan_spec
            >>> import delayed_image
            >>> delayed_image.SensorChanSpec = sensorchan_spec.SensorChanSpec  # hack for 3.6
            >>> a = delayed_image.SensorChanSpec.coerce('A|B|C,edf')
            >>> b = delayed_image.SensorChanSpec.coerce('A12')
            >>> c = delayed_image.SensorChanSpec.coerce('')
            >>> d = delayed_image.SensorChanSpec.coerce('rgb')
            >>> print(a.late_fuse(b).spec)
            >>> print((a + b).spec)
            >>> print((b + a).spec)
            >>> print((a + b + c).spec)
            >>> print(sum([a, b, c, d]).spec)
            A|B|C,edf,A12
            A|B|C,edf,A12
            A12,A|B|C,edf
            A|B|C,edf,A12
            A|B|C,edf,A12,rgb
            >>> import delayed_image
            >>> a = delayed_image.SensorChanSpec.coerce('A|B|C,edf').normalize()
            >>> b = delayed_image.SensorChanSpec.coerce('A12').normalize()
            >>> c = delayed_image.SensorChanSpec.coerce('').normalize()
            >>> d = delayed_image.SensorChanSpec.coerce('rgb').normalize()
            >>> print(a.late_fuse(b).spec)
            >>> print((a + b).spec)
            >>> print((b + a).spec)
            >>> print((a + b + c).spec)
            >>> print(sum([a, b, c, d]).spec)
            *:A|B|C,*:edf,*:A12
            *:A|B|C,*:edf,*:A12
            *:A12,*:A|B|C,*:edf
            *:A|B|C,*:edf,*:A12,*:
            *:A|B|C,*:edf,*:A12,*:,*:rgb
            >>> print((a.late_fuse(b)).concise())
            >>> print(((a + b)).concise())
            >>> print(((b + a)).concise())
            >>> print(((a + b + c)).concise())
            >>> print((sum([a, b, c, d])).concise())
            *:(A|B|C,edf,A12)
            *:(A|B|C,edf,A12)
            *:(A12,A|B|C,edf)
            *:(A|B|C,edf,A12,)
            *:(A|B|C,edf,A12,,r|g|b)

        Example:
            >>> # xdoctest: +REQUIRES(module:lark)
            >>> # Test multi-arg case
            >>> import delayed_image
            >>> a = delayed_image.SensorChanSpec.coerce('A|B|C,edf')
            >>> b = delayed_image.SensorChanSpec.coerce('A12')
            >>> c = delayed_image.SensorChanSpec.coerce('')
            >>> d = delayed_image.SensorChanSpec.coerce('rgb')
            >>> others = [b, c, d]
            >>> print(a.late_fuse(*others).spec)
            >>> print(delayed_image.SensorChanSpec.late_fuse(a, b, c, d).spec)
            A|B|C,edf,A12,rgb
            A|B|C,edf,A12,rgb
        """
        import itertools as it
        args = it.chain([self], others)
        specs = [s.spec for s in args if s.spec]
        new_spec = ','.join(specs)
        return SensorChanSpec.coerce(new_spec)

    def __add__(self, other):
        """
        Late fusion combination
        """
        return self.late_fuse(other)

    def __radd__(self, other):
        """
        Late fusion combination
        """
        if other == 0:
            return self
        return other.late_fuse(self)

    def matching_sensor(self, sensor):
        """
        Get the components corresponding to a specific sensor

        Args:
            sensor (str):
                the name of the sensor to match or "*" to match everything.

        Returns:
            FusedSensorChanSpec | SensorChanSpec: matching part of the spec

        Example:
            >>> # xdoctest: +REQUIRES(module:lark)
            >>> import delayed_image
            >>> self = delayed_image.SensorChanSpec.coerce('(S1,S2):(a|b|c),S2:c|d|e')
            >>> sensor = 'S2'
            >>> new = self.matching_sensor(sensor)
            >>> print(f'new={new}')
            new=S2:a|b|c,S2:c|d|e
            >>> print(self.matching_sensor('S1'))
            S1:a|b|c
            >>> print(self.matching_sensor('S3'))
            S3:
            >>> print(self.matching_sensor('*'))
            (S1,S2):(a|b|c),S2:c|d|e
        """
        # Handle special case
        if sensor == '*':
            return self

        # matching_streams = []
        # for s in self.streams():
        #     if s.sensor.spec == sensor:
        #         matching_streams.append(s)
        matching_streams = [
            s for s in self.streams()
            if s.sensor.spec == sensor or s.sensor.spec == '*'
        ]
        new = sum(matching_streams)
        if new == 0:
            import delayed_image
            new = FusedSensorChanSpec(SensorSpec(sensor), delayed_image.FusedChannelSpec.coerce(''))
        return new

    @property
    def chans(self):
        """
        Returns the channel-only spec, ONLY if all of the sensors are the same

        Example:
            >>> # xdoctest: +REQUIRES(module:lark)
            >>> import delayed_image
            >>> self = delayed_image.SensorChanSpec.coerce('(S1,S2):(a|b|c),S2:c|d|e')
            >>> import pytest
            >>> with pytest.raises(Exception):
            >>>     self.chans
            >>> print(self.matching_sensor('S1').chans.spec)
            >>> print(self.matching_sensor('S2').chans.spec)
            a|b|c
            a|b|c,c|d|e
        """
        channel_specs = []
        sensor_specs = []
        for s in self.streams():
            sensor_specs.append(s.sensor.spec)
            channel_specs.append(s.chans)
        if not ub.allsame(sensor_specs):
            raise Exception('Can only take pure channel specs when all sensors are the same')
        return sum(channel_specs)


class FusedSensorChanSpec(SensorChanSpec):
    """
    A single sensor a corresponding fused channels.

    Note:
        Using the explicit constructor should be avoided, as it will likely
        change for implementation efficiency in the future. Use `coerce` and
        `from_spec` instead. (Or submit a PR/MR with additional constructors
        you might find useful)

    Example:
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> from delayed_image.sensorchan_spec import *  # NOQA
        >>> assert FusedSensorChanSpec.coerce('sensor:a|b|c.0|c.1|c.2').spec == 'sensor:a|b|c.0|c.1|c.2'
        >>> assert FusedSensorChanSpec.coerce('a|b|c.0|c.1|c.2').spec == '*:a|b|c.0|c.1|c.2'
    """
    def __init__(self, sensor, chans):
        # Fixme, this signature does not agree with the parent
        self.sensor = sensor
        self._chans = chans

    @classmethod
    def from_spec(cls, spec: str) -> 'FusedSensorChanSpec':
        import delayed_image
        parts = sensorchan_normalized_parts(spec)
        if not len(parts) == 1:
            raise ValueError('must be a single fused set')
        node = parts[0]
        sensor = SensorSpec(node.sensor)
        chans = delayed_image.FusedChannelSpec.coerce(node.chan.spec)
        self = cls(sensor, chans)
        return self

    @classmethod
    def coerce(cls, data) -> 'FusedSensorChanSpec':
        """
        Attempt to interpret the data as a channel specification

        Returns:
            SensorChanSpec

        Example:
            >>> # xdoctest: +REQUIRES(module:lark)
            >>> from delayed_image.sensorchan_spec import *  # NOQA
            >>> from delayed_image.sensorchan_spec import FusedSensorChanSpec
            >>> self = FusedSensorChanSpec.coerce('*:u.0:3')
            >>> assert self.spec == '*:u.0|u.1|u.2'
        """
        if isinstance(data, cls):
            self = data
            return self
        elif isinstance(data, str):
            self = cls.from_spec(data)
            return self
        else:
            raise NotImplementedError

    def numel(self) -> int:
        return self.chans.numel()

    def normalize(self):
        # Fixme, not efficient
        new_spec = normalize_sensor_chan(self.spec)
        new = self.__class__.from_spec(new_spec)
        return new

    def concise(self):
        # Fixme, not efficient
        new_spec = concise_sensor_chan(self.spec)
        new = self.__class__.from_spec(new_spec)
        return new

    def split(self) -> 'SensorChanSpec':
        """
        Split each channel into a separate stream

        Returns:
            SensorChanSpec

        Example:
            >>> # xdoctest: +REQUIRES(module:lark)
            >>> from delayed_image import FusedSensorChanSpec
            >>> self = FusedSensorChanSpec.coerce('Cam1:(red|blue|feat.0:3)')
            >>> print(self.split().concise())
            Cam1:(red,blue,feat.0,feat.1,feat.2)
        """
        return super().split()

    @property
    def chans(self):
        return self._chans

    @property
    def spec(self):
        return '{}:{}'.format(self.sensor.spec, self.chans.spec)

    def __json__(self):
        return self.spec


class SensorChanNode:
    """
    TODO: just replace this with the spec class itself?
    """
    def __init__(self, sensor, chan):
        self.sensor = sensor
        self.chan = chan

    @property
    def spec(self):
        return f"{self.sensor}:{self.chan}"

    def __repr__(self):
        return self.spec

    def __str__(self):
        return self.spec


class FusedChanNode:
    """
    TODO: just replace this with the spec class itself?

    Example:
        s = FusedChanNode('a|b|c.0|c.1|c.2')
        c = s.concise()
        print(s)
        print(c)
    """
    def __init__(self, chan):
        import delayed_image
        self.data = delayed_image.FusedChannelSpec.coerce(chan)

    @property
    def spec(self):
        return self.data.spec

    def concise(self):
        return self.__class__(self.data.concise())

    def __repr__(self):
        return self.data.spec

    def __str__(self):
        return self.data.spec


class SensorChanTransformer(Transformer):
    """
    Given a parsed tree for a sensor-chan spec, can transform it into useful
    forms.

    TODO:
        Make the classes that hold the underlying data more robust such that
        they either use the existing channel spec or entirely replace it.
        (probably the former). Also need to add either a FusedSensorChan node
        that is restircted to only a single sensor and group of fused channels.

    Ignore:
        cases = [
             'S1:b:3',
             'S1:b:3,S2:b:3',
             'S1:b:3,S2:(b.0,b.1,b.2)',
        ]
        basis = {
            'concise_channels': [0, 1],
            'concise_sensors': [0, 1],
        }
        for spec in cases:
            print('')
            print('=====')
            print('spec = {}'.format(ub.urepr(spec, nl=1)))
            print('-----')
            for kwargs in ub.named_product(basis):
                sensor_channel_parser = _global_sensor_chan_parser()
                tree = sensor_channel_parser.parse(spec)
                transformed = SensorChanTransformer(**kwargs).transform(tree)
                print('')
                print('kwargs = {}'.format(ub.urepr(kwargs, nl=0)))
                print(f'transformed={transformed}')
            print('')
            print('=====')

    """

    def __init__(self, concise_channels=1, concise_sensors=1):
        self.consise_channels = concise_channels
        self.concise_sensors = concise_sensors

    def chan_id(self, items):
        code, = items
        return code.value

    def chan_single(self, items):
        code, = items
        return [code.value]

    def chan_getitem(self, items):
        code, index = items
        return [f'{code}.{index.value}']

    def chan_getslice_0b(self, items):
        code, btok = items
        return ['{}.{}'.format(code, index) for index in range(int(btok.value))]

    def chan_getslice_ab(self, items):
        code, atok, btok = items
        return ['{}.{}'.format(code, index) for index in range(int(atok.value), int(btok.value))]

    def chan_code(self, items):
        return items[0]

    def sensor_seq(self, items):
        return [s.value for s in items]

    def fused_seq(self, items):
        s = list(items)
        return s

    def fused(self, items):
        ret = FusedChanNode(list(ub.flatten(items)))
        if self.consise_channels:
            ret = ret.concise()
        return ret

    def channel_rhs(self, items):
        flat = []
        for item in items:
            if ub.iterable(item):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    def sensor_lhs(self, items):
        flat = []
        for item in items:
            if ub.iterable(item):
                flat.extend(item)
            else:
                flat.append(item.value)
        return flat

    def nosensor_chan(self, items):
        item, = items
        return [SensorChanNode('*', c) for c in item]

    def sensor_chan(self, items):
        if len(items) == 1:
            # handle empty channels
            items = [items[0], ['']]
        assert len(items) == 2
        lhs, rhs = items
        new = []
        for a, b in it.product(lhs, rhs):
            new.append(SensorChanNode(a, b))
        return new

    def stream_item(self, items):
        item, = items
        return item

    def stream(self, items):
        flat_items = list(ub.flatten(items))
        # TODO: can probably improve this
        if self.concise_sensors:
            flat_sensors = [str(f.sensor) for f in flat_items]
            flat_chans = [str(f.chan) for f in flat_items]
            chan_to_sensors = ub.group_items(flat_sensors, flat_chans)

            pass1_sensors = []
            pass1_chans = []
            for chan, sensors in chan_to_sensors.items():
                sense_part = ','.join(sorted(ub.unique(sensors)))
                if len(sensors) > 1:
                    sense_part = '({})'.format(sense_part)
                pass1_sensors.append(sense_part)
                pass1_chans.append(str(chan))

            pass2_parts = []
            sensor_to_chan = ub.group_items(pass1_chans, pass1_sensors)
            for sensor, chans in sensor_to_chan.items():
                chan_part = ','.join(chans)
                if len(chans) > 1:
                    chan_part = '({})'.format(chan_part)
                pass2_parts.append('{}:{}'.format(sensor, chan_part))

            parts = pass2_parts
            parts = sorted(parts)
        else:
            parts = flat_items
        return parts


@cache
def _global_sensor_chan_parser():
    # https://github.com/lark-parser/lark/blob/master/docs/_static/lark_cheatsheet.pdf
    import lark
    try:
        import lark_cython
        sensor_channel_parser = lark.Lark(SENSOR_CHAN_GRAMMAR,  start='start', parser='lalr', _plugins=lark_cython.plugins)
    except ImportError:
        sensor_channel_parser = lark.Lark(SENSOR_CHAN_GRAMMAR,  start='start', parser='lalr')
    return sensor_channel_parser


@cache
def normalize_sensor_chan(spec: str):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> from delayed_image.sensorchan_spec import *  # NOQA
        >>> spec = 'L8:mat:4,L8:red,S2:red,S2:forest|brush,S2:mat.0|mat.1|mat.2|mat.3'
        >>> r1 = normalize_sensor_chan(spec)
        >>> spec = 'L8:r|g|b,L8:r|g|b'
        >>> r2 = normalize_sensor_chan(spec)
        >>> print(f'r1={r1}')
        >>> print(f'r2={r2}')
        r1=L8:mat.0|mat.1|mat.2|mat.3,L8:red,S2:red,S2:forest|brush,S2:mat.0|mat.1|mat.2|mat.3
        r2=L8:r|g|b,L8:r|g|b

    Ignore:
        >>> # TODO: fix bug or disallow behavior
        >>> from delayed_image.sensorchan_spec import *  # NOQA
        >>> spec = '*:(rgb,,cde)'
        >>> concise_spec = normalize_sensor_chan(spec)
    """
    if spec == '':
        spec = '*:'
    transformed = sensorchan_normalized_parts(spec)
    new_spec = ','.join([n.spec for n in transformed])
    return new_spec


@cache
def concise_sensor_chan(spec: str):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> from delayed_image.sensorchan_spec import *  # NOQA
        >>> spec = 'L8:mat.0|mat.1|mat.2|mat.3,L8:red,S2:red,S2:forest|brush,S2:mat.0|mat.1|mat.2|mat.3'
        >>> concise_spec = concise_sensor_chan(spec)
        >>> normed_spec = normalize_sensor_chan(concise_spec)
        >>> concise_spec2 = concise_sensor_chan(normed_spec)
        >>> assert concise_spec2 == concise_spec
        >>> print(concise_spec)
        (L8,S2):(mat:4,red),S2:forest|brush
    """
    transformed = sensorchan_concise_parts(spec)
    new_spec = ','.join([str(n) for n in transformed])
    return new_spec


# @cache
def sensorchan_concise_parts(spec: str):
    """
    Ignore:
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> spec = 'L8:mat.0|mat.1|mat.2|mat.3,L8:red,(MODIS,S2):a|b|c,S2:red,S2:forest|brush|bare_ground,S2:mat.0|mat.1|mat.2|mat.3'
        >>> parts = sensorchan_concise_parts(spec)
    """
    try:
        sensor_channel_parser = _global_sensor_chan_parser()
        tree = sensor_channel_parser.parse(spec)
        transformed = SensorChanTransformer(concise_sensors=1, concise_channels=1).transform(tree)
    except Exception:
        print(f'ERROR: Failed to condense spec={spec}')
        raise
    return transformed


def sensorchan_normalized_parts(spec: str):
    """
    Ignore:
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> spec = 'L8:mat.0|mat.1|mat.2|mat.3,L8:red,(MODIS,S2):a|b|c,S2:red,S2:forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,S2:mat.0|mat.1|mat.2|mat.3'
        >>> parts = sensorchan_normalized_parts(spec)
    """
    try:
        sensor_channel_parser = _global_sensor_chan_parser()
        tree = sensor_channel_parser.parse(spec)
        transformed = SensorChanTransformer(concise_sensors=0, concise_channels=0).transform(tree)
    except Exception:
        print(f'ERROR: Failed to normalize spec={spec}')
        raise
    return transformed
