import { Image, ImageFinderInterface, imageResource, MatchRequest, MatchResult, Region, screen } from '@nut-tree/nut-js';
import { ScaleImageHandler } from './handlers/scaleImage';
import { ImageProcessor } from './readers/imageProcessor.class';
import { Mat } from 'opencv4nodejs-prebuilt-install/lib/typings/Mat';
import { CustomMatchRequest, MethodEnum, MethodNameType, CustomConfigType, SingleTargetMatch } from './types';
import { OverWritingMatcherHandler } from './handlers/overWriting';
import { ValidationHandler } from './handlers/validation';
import { NonMaximumSuppressionHandler } from './handlers/nonMaximumSuppression';
import { InvariantRotatingHandler } from './handlers/invariantRotating';
import { imshow, Point2, Rect, waitKey } from 'opencv4nodejs-prebuilt-install';

export default class TemplateMatchingFinder implements ImageFinderInterface {
  private _config: CustomConfigType;

  constructor() {
    this._config = {
      confidence: 0.8,
      providerData: {
        scaleSteps: [1, 0.9, 0.8, 0.7, 0.6, 0.5],
        methodType: MethodEnum.TM_CCOEFF_NORMED,
        debug: false,
        isSearchMultipleScales: true,
        isRotation: false,
        rotationOption: { range: 180, overLap: 0.1, minDstLength: 256 },
      },
    };
  }

  getConfig() {
    return this._config;
  }

  setConfig(config: CustomConfigType) {
    this._config = { ...this._config, ...config };
  }

  private async loadNeedle(image: Image | string): Promise<{ data: Mat }> {
    if (typeof image !== 'string') {
      return { data: await ImageProcessor.fromImageWithAlphaChannel(image) };
    } else {
      return { data: await ImageProcessor.fromImageWithAlphaChannel(await imageResource(image)) };
    }
  }

  private async loadHaystack(
    image?: Image | string,
    roi?: Region,
  ): Promise<{
    data: Mat;
    rect: Region | undefined;
    pixelDensity: {
      scaleX: number;
      scaleY: number;
    };
  }> {
    if (typeof image !== 'string' && image) {
      let validRoi = roi ? ValidationHandler.determineMatRectROI(image, ValidationHandler.getIncreasedRectByPixelDensity(roi, image.pixelDensity)) : undefined;

      return {
        data: await ImageProcessor.fromImageWithAlphaChannel(image, validRoi),
        rect: validRoi ? ValidationHandler.determineRegionRectROI(validRoi) : undefined,
        pixelDensity: image.pixelDensity,
      };
    } else {
      if (!image) {
        const imageObj = await screen.grab();
        let validRoi = roi ? ValidationHandler.determineMatRectROI(imageObj, ValidationHandler.getIncreasedRectByPixelDensity(roi, imageObj.pixelDensity)) : undefined;
        const mat = await ImageProcessor.fromImageWithAlphaChannel(imageObj, validRoi);

        return { data: mat, rect: validRoi ? ValidationHandler.determineRegionRectROI(validRoi) : undefined, pixelDensity: imageObj.pixelDensity };
      } else {
        const imageObj = await imageResource(image);
        let validRoi = roi ? ValidationHandler.determineMatRectROI(imageObj, ValidationHandler.getIncreasedRectByPixelDensity(roi, imageObj.pixelDensity)) : undefined;

        return {
          data: await ImageProcessor.fromImageWithAlphaChannel(imageObj, validRoi),
          rect: validRoi ? ValidationHandler.determineRegionRectROI(validRoi) : undefined,
          pixelDensity: imageObj.pixelDensity,
        };
      }
    }
  }

  private async initData<OptionalSearchParameters>(matchRequest: MatchRequest<Image, OptionalSearchParameters> | CustomMatchRequest) {
    const customMatchRequest = matchRequest as CustomMatchRequest;
    const confidence =
      customMatchRequest.providerData && customMatchRequest.providerData?.methodType === MethodEnum.TM_SQDIFF && matchRequest.confidence === 0.99
        ? 0.998
        : (customMatchRequest.providerData && customMatchRequest.providerData?.methodType === MethodEnum.TM_CCOEFF_NORMED) ||
          (customMatchRequest.providerData && customMatchRequest.providerData?.methodType === MethodEnum.TM_CCORR_NORMED && matchRequest.confidence === 0.99)
        ? (this._config.confidence as number)
        : matchRequest.confidence === 0.99 || typeof matchRequest.confidence === 'undefined'
        ? (this._config.confidence as number)
        : matchRequest.confidence;
    const isSearchMultipleScales =
      customMatchRequest.providerData && 'scaleSteps' in customMatchRequest.providerData && customMatchRequest.providerData.scaleSteps?.length ? true : !!this._config.providerData?.scaleSteps?.length;
    const scaleSteps = customMatchRequest.providerData?.scaleSteps || (this._config.providerData?.scaleSteps as Array<number>);
    const methodType = customMatchRequest.providerData?.methodType || (this._config.providerData?.methodType as MethodNameType);
    const debug = customMatchRequest.providerData?.debug || (this._config.providerData?.debug as boolean);
    const isRotation = customMatchRequest.providerData?.isRotation || this._config.providerData?.isRotation;
    const rotationOverLap = customMatchRequest.providerData?.rotationOption?.overLap || (this._config.providerData?.rotationOption?.overLap as number);
    const rotationRange = customMatchRequest.providerData?.rotationOption?.range || this._config.providerData?.rotationOption?.range;
    const rotationMinLength = customMatchRequest.providerData?.rotationOption?.minDstLength || this._config.providerData?.rotationOption?.minDstLength;

    const needle = await this.loadNeedle(matchRequest.needle);
    if (!needle || needle.data.empty) {
      throw new Error(`Failed to load ${typeof matchRequest.needle === 'string' ? matchRequest.needle : matchRequest.needle.id}, got empty image.`);
    }
    const haystack = await this.loadHaystack(matchRequest.haystack, customMatchRequest.providerData?.roi);
    if (!haystack || haystack.data.empty) {
      throw new Error(
        `Failed to load ${
          matchRequest && matchRequest.haystack && typeof matchRequest.haystack === 'string' && !matchRequest.haystack ? matchRequest.haystack : (matchRequest.haystack as Image).id
        }, got empty image.`,
      );
    }
    if (isSearchMultipleScales) {
      ValidationHandler.throwOnTooLargeNeedle(haystack.data, needle.data, scaleSteps[scaleSteps.length - 1]);
    }

    return {
      haystack: haystack,
      needle: needle,
      confidence: confidence,
      scaleSteps: scaleSteps,
      methodType: methodType,
      debug: debug,
      isSearchMultipleScales: isSearchMultipleScales,
      roi: customMatchRequest.providerData?.roi,
      isRotation: isRotation,
      rotationOverLap: rotationOverLap,
      rotationRange: rotationRange,
      rotationMinLength: rotationMinLength,
    };
  }

  public async findMatch<OptionalSearchParameters>(matchRequest: MatchRequest<Image, OptionalSearchParameters> | CustomMatchRequest): Promise<MatchResult<Region>> {
    let { haystack, rotationOverLap, rotationRange, isRotation, rotationMinLength, needle, confidence, scaleSteps, methodType, debug, isSearchMultipleScales, roi } = await this.initData(matchRequest);
    let matchResults: Array<MatchResult<Region>> = [];

    if (!isSearchMultipleScales) {
      if (isRotation) {
        const rotatedResults = await InvariantRotatingHandler.Match(haystack.data, needle.data, rotationMinLength as number, confidence, rotationRange, rotationOverLap);
        matchResults = this.getRotatedFullRectanglePointWithoutAngle(rotatedResults).map(
          (i) => new MatchResult<Region>(i.match.dMatchScore, new Region(i.point.x, i.point.y, i.newSize.width, i.newSize.height)),
        );
      } else {
        const matches = await OverWritingMatcherHandler.matchImages(haystack.data, needle.data, methodType, debug);
        matchResults = [matches.data];
      }
      return (await ValidationHandler.getValidatedMatches(matchResults, haystack.pixelDensity, confidence, roi))[0];
    } else {
      let matchResults: Array<MatchResult<Region>> = [];

      if (isRotation) {
        const rotatedResults = await InvariantRotatingHandler.Match(haystack.data, needle.data, rotationMinLength as number, confidence, rotationRange, rotationOverLap);
        matchResults = this.getRotatedFullRectanglePointWithoutAngle(rotatedResults).map(
          (i) => new MatchResult<Region>(i.match.dMatchScore, new Region(i.point.x, i.point.y, i.newSize.width, i.newSize.height)),
        );
      } else {
        matchResults = await ScaleImageHandler.searchMultipleScales(haystack.data, needle.data, confidence, scaleSteps, methodType, debug, true);
      }
      return (await ValidationHandler.getValidatedMatches(matchResults.length ? [matchResults[0]] : matchResults, haystack.pixelDensity, confidence, roi))[0];
    }
  }

  public async findMatches<OptionalSearchParameters>(matchRequest: MatchRequest<Image, OptionalSearchParameters> | CustomMatchRequest): Promise<MatchResult<Region>[]> {
    let matchResults: Array<MatchResult<Region>> = [];
    let { haystack, rotationOverLap, rotationRange, rotationMinLength, needle, confidence, scaleSteps, methodType, debug, isSearchMultipleScales, roi } = await this.initData(matchRequest);

    if (!isSearchMultipleScales) {
      if (rotationRange) {
        const rotatedResults = await InvariantRotatingHandler.Match(haystack.data, needle.data, rotationMinLength as number, confidence, rotationRange, rotationOverLap);
        matchResults = this.getRotatedFullRectanglePointWithoutAngle(rotatedResults).map(
          (i) => new MatchResult<Region>(i.match.dMatchScore, new Region(i.point.x, i.point.y, i.newSize.width, i.newSize.height)),
        );
      } else {
        const overwrittenResults = await OverWritingMatcherHandler.matchImagesByWriteOverFounded(haystack.data, needle.data, confidence, methodType, debug);
        matchResults.push(...overwrittenResults.results);
      }
    } else {
      if (rotationRange) {
        const rotatedResults = await InvariantRotatingHandler.Match(haystack.data, needle.data, rotationMinLength as number, confidence, rotationRange, rotationOverLap);
        matchResults = this.getRotatedFullRectanglePointWithoutAngle(rotatedResults).map(
          (i) => new MatchResult<Region>(i.match.dMatchScore, new Region(i.point.x, i.point.y, i.newSize.width, i.newSize.height)),
        );
      } else {
        const scaledResults = await ScaleImageHandler.searchMultipleScales(haystack.data, needle.data, confidence, scaleSteps, methodType, debug);
        matchResults.push(...scaledResults);
      }
    }
    const suppressedMatchResults = NonMaximumSuppressionHandler.filterMatchResult(matchResults);

    return await ValidationHandler.getValidatedMatches(suppressedMatchResults, haystack.pixelDensity, confidence, roi);
  }

  private getRotatedFullRectanglePointWithoutAngle(matches: Array<SingleTargetMatch>) {
    const points = [];

    for (let match of matches) {
      const veryTopPoint_y = Math.min(match.ptLB.y, match.ptLT.y, match.ptRB.y, match.ptRT.y);
      const veryBottomPoint_y = Math.max(match.ptLB.y, match.ptLT.y, match.ptRB.y, match.ptRT.y);
      const veryLeftPoint_x = Math.min(match.ptLB.x, match.ptLT.x, match.ptRB.x, match.ptRT.x);
      const veryRightPoint_x = Math.max(match.ptLB.x, match.ptLT.x, match.ptRB.x, match.ptRT.x);

      const getPointY = (axis: number) => {
        return Object.entries(match)
          .map((p) => {
            if ((p[0] === 'ptLB' || p[0] === 'ptLT' || p[0] === 'ptRB' || p[0] === 'ptRT') && p[1].y === axis) {
              return { x: p[1].x, y: p[1].y };
            } else {
              return undefined;
            }
          })
          .filter((f) => f !== undefined)[0] as { x: number; y: number };
      };

      const getPointX = (axis: number) => {
        return Object.entries(match)
          .map((p) => {
            if ((p[0] === 'ptLB' || p[0] === 'ptLT' || p[0] === 'ptRB' || p[0] === 'ptRT') && p[1].x === axis) {
              return { x: p[1].x, y: p[1].y };
            } else {
              return undefined;
            }
          })
          .filter((f) => f !== undefined)[0] as { x: number; y: number };
      };

      let veryTopPoint: { x: number; y: number } = getPointY(veryTopPoint_y);
      let veryBottomPoint: { x: number; y: number } = getPointY(veryBottomPoint_y);
      let veryLeftPoint: { x: number; y: number } = getPointX(veryLeftPoint_x);
      let veryRightPoint: { x: number; y: number } = getPointX(veryRightPoint_x);

      const accurateTopPoint = veryTopPoint.x > veryLeftPoint.x ? veryLeftPoint.x : veryTopPoint.x;
      const offsetTopPointWidth = veryRightPoint.x - accurateTopPoint > match.size.width ? veryRightPoint.x - accurateTopPoint : match.size.width;
      const offsetTopPointHeight = veryBottomPoint.y - veryTopPoint.y > match.size.height ? veryBottomPoint.y - veryTopPoint.y : match.size.height;

      const newSize: { width: number; height: number } = { width: offsetTopPointWidth, height: offsetTopPointHeight };
      points.push({ match: match, point: new Point2(accurateTopPoint, veryTopPoint.y), newSize: newSize });
    }
    return points;
  }
}
