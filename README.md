# OpenCV 4.1.1 Template Matching Image Finder

![Tested](https://github.com/udarrr/TemplateMatcher/workflows/Tests/badge.svg)
![Released](https://github.com/udarrr/TemplateMatcher/workflows/Create%20tagged%20release/badge.svg)
![Supported node LTS versions](https://img.shields.io/badge/node@arch64-12%2C%2013%2C%2014%2C%2015%2C%2016%2C%2017%2C%2018%2C%2019-green)

## It's either standalone or plugin for [nutjs project](https://www.npmjs.com/package/@nut-tree/nut-js)

### Installation for standalone

```nodejs
npm i @udarrr/template-matcher
```

### Installation for nutjs

```nodejs
npm i @udarrr/template-matcher
```

### Options findMatch,findMatches

#### Standalone

```typescript
{
    haystack?: string | Image,
    needle: string | Image,
    confidence?: number,
    searchMultipleScales?: boolean,
    customOptions?: {
                       methodType?: MethodNameType; 
                       scaleSteps?: Array<number>; 
                       roi?: Region; 
                       debug?: boolean
                    },
}
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

### Nutjs v3

```nodejs
npm i @udarrr/template-matcher@2.0.4
```

```typescript
{
  providerData?: {
    searchMultipleScales?: boolean;
    methodType?: MethodNameType;
    scaleSteps?: Array<number>;
    roi?: Region;
    debug?: boolean;
  };
};
```

```typescript
import { imageResource, screen } from '@nut-tree/nut-js';
import { CustomOptionsType } from '@udarrr/template-matcher/lib/template-matching-finder.class';
import "@udarrr/template-matcher"; //once wherever

(async () => {
  const img = await screen.find<CustomOptionsType>(imageResource("path"),{ providerData: {...}});
  const imgs = await screen.findAll<CustomOptionsType>(imageResource("path"),{ providerData: {...}});
})();

```

#### Values by default

```typescript
methodType: "TM_CCOEFF" | "TM_CCOEFF_NORMED" | "TM_CCORR" | "TM_CCORR_NORMED" | "TM_SQDIFF" | "TM_SQDIFF_NORMED" by default "TM_CCOEFF_NORMED"
scaleSteps:  [0.9]; by default  [1, 0.9, 0.8, 0.7, 0.6, 0.5]
debug: true | false by default false
confidence: for "TM_SQDIFF" | "TM_SQDIFF_NORMED" confidence by default 0.98 for "TM_CCOEFF" | "TM_CCOEFF_NORMED" | "TM_CCORR" | "TM_CCORR_NORMED" by default 0.8
```

#### Disclaimer for nutjs v3

In case using the package with [nutjs](https://github.com/nut-tree/nut.js/blob/develop/lib/optionalsearchparameters.class.ts) v3 or above please use precise 2.0.2 version of the package it could be prevented in package.json lile

```json
 "@udarrr/template-matcher": "~2.0.4",
```
