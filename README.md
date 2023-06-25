# OpenCV 4.1.1 Template Matching Image Finder

![Tested](https://github.com/udarrr/TemplateMatcher/workflows/Tests/badge.svg)
![Released](https://github.com/udarrr/TemplateMatcher/workflows/Create%20tagged%20release/badge.svg)
![Supported node LTS versions](https://img.shields.io/badge/node@arch64-12%2C%2013%2C%2014%2C%2015%2C%2016%2C%2017%2C%2018%2C%2019%2C%2020-green)

## It's either standalone or plugin for [nutjs project](https://www.npmjs.com/package/@nut-tree/nut-js)

The best template matcher for node js ever with handlers

- Invariant Rotating
- Over Writing
- Scale Images
- Non Maximum Suppression

### Standalone findMatch,findMatches

```nodejs
npm i @udarrr/template-matcher
```

```typescript
import finder from "@udarrr/template-matcher";

(async () => {
const matcheImages = await finder.findMatch({haystack: 'pathToImage', needle: 'pathToTemplate'});
const matcheWithScreen = await finder.findMatch({needle: pathToTemplate});

const matchesImages = await finder.findMatches({haystack: 'pathToImage', needle: 'pathToTemplate'});
const matchesWithScreen = await finder.findMatches({needle: 'pathToTemplate'});
})();

```

#### @udarrr/template-matcher standalone API

```typescript
{
    haystack?: string | Image,
    needle: string | Image,
    confidence?: number,
    providerData?: {
                       methodType?: MethodNameType; 
                       scaleSteps?: Array<number>; 
                       searchMultipleScales: boolean,
                       isRotation: boolean,
                       rotationOption?: { range?: number; overLap?: number; minDstLength?: number };
                       roi?: Region; 
                       debug?: boolean
                    },
}
```

### Nutjs v3 find,findAll

```nodejs
npm i @udarrr/template-matcher
```

```typescript
import { imageResource, screen } from '@nut-tree/nut-js';
import {OptionsSearchParameterType} from '@udarrr/template-matcher/lib/types'
import "@udarrr/template-matcher"; //once wherever

(async () => {
  const img = await screen.find<OptionsSearchParameterType>(imageResource("path"),{ providerData: {...}});
  const imgs = await screen.findAll<OptionsSearchParameterType>(imageResource("path"),{ providerData: {...}});
})();

```

#### @udarrr/template-matcher providerData nutjs v3 Api

```typescript
{
  providerData?: {
      methodType?: MethodNameType;
      scaleSteps?: Array<number>;
      searchMultipleScales: boolean,
      isRotation: boolean,
      rotationOption?: { range?: number; overLap?: number; minDstLength?: number };
      debug?: boolean;
      roi?: Region;
  }
};
```

#### Values by default

```typescript
methodType: "TM_CCOEFF_NORMED"
scaleSteps: [1, 0.9, 0.8, 0.7, 0.6, 0.5]
debug: false
searchMultipleScales: true,
isRotation: false,
rotationOption: {range: 180, overLap: 0.1, minDstLength: 256}
confidence: 0.8
```
