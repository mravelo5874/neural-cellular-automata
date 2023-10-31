from struct import unpack_from as unpack, calcsize
from VoxModels import VoxObj, Size, Voxel, Color, Model, Material

def bit(val, offset):
    mask = 1 << offset
    return(val & mask)

class ParsingException(Exception): pass

class Chunk(object):
    def __init__(self, id, content=None, chunks=None):
        self.id = id
        self.content = content or b''
        self.chunks = chunks or []
        self.material = None

        if id == b'MAIN':
            if len(self.content): raise ParsingException('Non-empty content for main chunk')
        elif id == b'PACK':
            self.models = unpack('i', content)[0]
        elif id == b'SIZE':
            self.size = Size(*unpack('iii', content))
        elif id == b'XYZI':
            n = unpack('i', content)[0]
            #print(f'xyzi block with {n} voxels (len {len(content)})')
            self.voxels = []
            self.voxels = [ Voxel(*unpack('BBBB', content, 4+4*i)) for i in range(n) ]
        elif id == b'RGBA':
            self.palette = [ Color(*unpack('BBBB', content, 4*i)) for i in range(255) ]
            # Docs say:  color [0-254] are mapped to palette index [1-255]
            # hmm
            # self.palette = [ Color(0,0,0,0) ] + [ Color(*unpack('BBBB', content, 4*i)) for i in range(255) ]
        elif id == b'MATT':
            _id, _type, weight, flags = unpack('iifi', content)
            props = {}
            offset = 16
            for b,field in [ (0, 'plastic'),
                             (1, 'roughness'),
                             (2, 'specular'),
                             (3, 'IOR'),
                             (4, 'attenuation'),
                             (5, 'power'),
                             (6, 'glow'),
                             (7, 'isTotalPower') ]:
                if bit(flags, b) and b<7: # no value for 7 / isTotalPower
                    props[field] = unpack('f', content, offset)
                    offset += 4
            self.material = Material(_id, _type, weight, props)
            
        elif id == b'nTRN':
            pass
            #print ('found nTRN chunk!')
        elif id == b'nGRP':
            pass
            #print ('found nGRP chunk!')
        elif id == b'nSHP':
            pass
            #print ('found nSHP chunk!')
        elif id == b'MATL':
            pass
            #print ('found MATL chunk!')
        elif id == b'LAYR':
            pass
            #print ('found LAYR chunk!')
        elif id == b'rOBJ':
            pass
            #print ('found rOBJ chunk!')
        elif id == b'rCAM':
            pass
            #print ('found rCAM chunk!')
        elif id == b'NOTE':
            pass
            #print ('found NOTE chunk!')
        elif id == b'IMAP':
            pass
            #print ('found IMAP chunk!')
        else:
            raise ParsingException('Unknown chunk type: %s'%self.id)
        
class voxparser(object):
    def __init__(self, _filename):
        with open(_filename, 'rb') as f:
            self.content = f.read()
        self.offset = 0
        
    def unpack(self, fmt):
        r = unpack(fmt, self.content, self.offset)
        self.offset += calcsize(fmt)
        return r

    def _parseChunk(self):
        _id, N, M = self.unpack('4sii')
        #print(f'Found chunk id {_id} / len {N} / children {M}')
        content = self.unpack('%ds'%N)[0]
        start = self.offset
        chunks = [ ]
        while self.offset<start+M:
            chunks.append(self._parseChunk())

        return Chunk(_id, content, chunks)
    
    def _parseModel(self, size, xyzi):
        if size.id != b'SIZE': raise ParsingException('Expected SIZE chunk, got %s', size.id)
        if xyzi.id != b'XYZI': raise ParsingException('Expected XYZI chunk, got %s', xyzi.id)

        return Model(size.size, xyzi.voxels)
            
    def parse(self):
        _, version = self.unpack('4si')
        
        if version != 200 and version != 150: raise ParsingException("Unknown vox version: %s expected 150"%version)
        
        main = self._parseChunk()
        if main.id != b'MAIN': raise ParsingException("Missing MAIN Chunk")

        chunks = list(reversed(main.chunks))
        if chunks[-1].id == b'PACK':
            models = chunks.pop().models
        else:
            models = 1

        models = [self._parseModel(chunks.pop(), chunks.pop()) for _ in range(models)]

        if chunks and chunks[0].id == b'RGBA':
            palette = chunks.pop().palette
        else:
            palette = None

        materials = [ c.material for c in chunks ]

        return VoxObj(models, palette, materials)