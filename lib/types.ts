import { Image, MatchResult, Region } from '@nut-tree/nut-js';
import { Mat } from 'opencv4nodejs-prebuilt-install/lib/typings/Mat';
import { Point2 } from 'opencv4nodejs-prebuilt-install/lib/typings/Point2';
import { Rect } from 'opencv4nodejs-prebuilt-install/lib/typings/Rect';
import { RotatedRect } from 'opencv4nodejs-prebuilt-install/lib/typings/RotatedRect';
import { Size } from 'opencv4nodejs-prebuilt-install/lib/typings/Size';

export enum MethodEnum {
  TM_CCOEFF = 'TM_CCOEFF',
  TM_CCOEFF_NORMED = 'TM_CCOEFF_NORMED',
  TM_CCORR = 'TM_CCORR',
  TM_CCORR_NORMED = 'TM_CCORR_NORMED',
  TM_SQDIFF = 'TM_SQDIFF',
  TM_SQDIFF_NORMED = 'TM_SQDIFF_NORMED',
}
export type MethodNameType = 'TM_CCOEFF' | 'TM_CCOEFF_NORMED' | 'TM_CCORR' | 'TM_CCORR_NORMED' | 'TM_SQDIFF' | 'TM_SQDIFF_NORMED';
export type MatchedResults = { results: Array<MatchResult<Region>>; haystack: Mat };

export interface OptionsSearchParameterType {
  methodType?: MethodNameType;
  scaleSteps?: Array<number>;
  searchMultipleScales?: boolean;
  isRotation?: boolean;
  rotationOption?: { range?: number; overLap?: number; minDstLength?: number; subPixEstimation?: boolean };
  debug?: boolean;
  roi?: Region;
}
export interface CustomConfigType {
  confidence?: number;
  providerData?: OptionsSearchParameterType;
}

export interface CustomMatchRequest extends CustomConfigType {
  haystack?: Image | string;
  needle: Image | string;
}

export interface Vector {
  vecPyramid: Array<Mat>;
  vecTemplMean: Array<Array<number>>;
  vecTemplNorm: Array<number>;
  vecInvArea: Array<number>;
  vecResultEqual1: Array<boolean>;
  bIsPatternLearned: boolean;
  iBorderColor: number;
}

export interface SingleTargetMatch {
  ptLT: Point2;
  ptRT: Point2;
  ptRB: Point2;
  ptLB: Point2;
  ptCenter: Point2;
  dMatchedAngle: number;
  dMatchScore: number;
  size: Size;
}

export interface MatchParameter {
  pt: Point2;
  dMatchScore: number;
  dMatchAngle: number;
  matRotatedSrc?: Mat;
  rectRoi?: Rect;
  dAngleStart?: number;
  dAngleEnd?: number;
  rectR?: RotatedRect;
  rectBounding?: Rect;
  bDelete?: boolean;
  size: Size;

  vecResult?: number[][];
  iMaxScoreIndex?: number;
  bPosOnBorder?: boolean;
  ptSubPixel?: Rect;
  dNewAngle?: number;
}
