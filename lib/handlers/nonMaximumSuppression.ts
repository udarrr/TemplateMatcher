import { MatchResult, Region } from '@nut-tree-fork/nut-js';

const { min, max } = Math;

export class NonMaximumSuppressionHandler {
  static filterMatchResult(foundLocations: Array<MatchResult<Region>>, overlapThresh: number = 0.4): MatchResult<Region>[] {
    let possibleLocation: Array<{ x1: number; x2: number; y1: number; y2: number; width: number; height: number; area: number }> = [];
    let suppress: Array<{ x1: number; x2: number; y1: number; y2: number; width: number; height: number; area: number }> = [];

    if (foundLocations.length === 0) {
      return [];
    }

    const pick: Array<{ x1: number; x2: number; y1: number; y2: number; width: number; height: number; area: number }> = [];

    possibleLocation = foundLocations.map((rect) => {
      return {
        x1: rect.location.left,
        y1: rect.location.top,
        x2: rect.location.left + rect.location.width,
        y2: rect.location.top + rect.location.height,
        width: rect.location.width,
        height: rect.location.height,
        area: (rect.location.height + 1) * (rect.location.width + 1),
      };
    });

    possibleLocation.sort((b1, b2) => {
      return b1.y2 - b2.y2;
    });

    while (possibleLocation.length > 0) {
      let last = possibleLocation[possibleLocation.length - 1];
      pick.push(last);
      suppress = [last];

      for (let i = 0; i < possibleLocation.length - 1; i++) {
        const xx1 = max(possibleLocation[i].x1, last.x1);
        const yy1 = max(possibleLocation[i].y1, last.y1);
        const xx2 = min(possibleLocation[i].x2, last.x2);
        const yy2 = min(possibleLocation[i].y2, last.y2);
        const w = max(0, xx2 - xx1 + 1);
        const h = max(0, yy2 - yy1 + 1);
        const overlap = (w * h) / possibleLocation[i].area;

        if (overlap > overlapThresh) {
          suppress.push(possibleLocation[i]);
        }
      }

      possibleLocation = possibleLocation.filter((box) => {
        return !suppress.find((supp) => {
          return supp === box;
        });
      });
    }
    return foundLocations.filter((rect: MatchResult<Region>) => pick.some((i) => i.x1 === rect.location.left && i.y1 === rect.location.top));
  }
}
