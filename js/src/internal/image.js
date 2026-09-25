// Decoding encoded image bytes to RGB for multimodal input.

import { toUint8Array } from './typed_values.js';

export async function decodeImageBytesToRgb(bytes, options = {}) {
  const sourceBytes = toUint8Array(bytes);
  if (!sourceBytes || sourceBytes.length === 0) {
    return null;
  }

  if (typeof createImageBitmap !== 'function' || typeof Blob !== 'function') {
    return null;
  }

  const maxPixelsCandidate = Number(options.maxPixels);
  const maxPixels = Number.isFinite(maxPixelsCandidate) && maxPixelsCandidate > 0
    ? Math.max(65536, Math.min(33554432, Math.trunc(maxPixelsCandidate)))
    : 0;
  const maxEdgeCandidate = Number(options.maxEdge);
  const maxEdge = Number.isFinite(maxEdgeCandidate) && maxEdgeCandidate > 0
    ? Math.max(64, Math.min(16384, Math.trunc(maxEdgeCandidate)))
    : 0;

  if (maxPixels <= 0 && maxEdge <= 0) {
    return null;
  }

  let bitmap = null;
  try {
    const mimeType =
      typeof options.mimeType === 'string' && options.mimeType.length > 0
        ? options.mimeType
        : 'image/png';
    const blob = new Blob([/** @type {BlobPart} */ (sourceBytes)], { type: mimeType });
    bitmap = await createImageBitmap(blob);

    const sourceWidth = Math.max(1, Math.trunc(Number(bitmap.width) || 0));
    const sourceHeight = Math.max(1, Math.trunc(Number(bitmap.height) || 0));

    let scale = 1;
    if (maxPixels > 0) {
      const sourcePixels = sourceWidth * sourceHeight;
      if (sourcePixels > maxPixels) {
        scale = Math.min(scale, Math.sqrt(maxPixels / sourcePixels));
      }
    }
    if (maxEdge > 0) {
      const sourceLongest = Math.max(sourceWidth, sourceHeight);
      if (sourceLongest > maxEdge) {
        scale = Math.min(scale, maxEdge / sourceLongest);
      }
    }

    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(1, Math.round(sourceHeight * scale));

    let canvas = null;
    let context = null;
    if (typeof OffscreenCanvas === 'function') {
      canvas = new OffscreenCanvas(width, height);
      context = canvas.getContext('2d', {
        alpha: false,
        willReadFrequently: true,
      });
    }

    if (!context && typeof document !== 'undefined' && typeof document.createElement === 'function') {
      canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      context = canvas.getContext('2d', {
        alpha: false,
        willReadFrequently: true,
      });
    }

    if (!context) {
      return null;
    }

    context.drawImage(bitmap, 0, 0, width, height);

    let encodedBytes = null;
    if (canvas && 'convertToBlob' in canvas && typeof canvas.convertToBlob === 'function') {
      const encodedBlob = await canvas.convertToBlob({ type: 'image/png' });
      if (encodedBlob) {
        encodedBytes = new Uint8Array(await encodedBlob.arrayBuffer());
      }
    }

    if (
      !encodedBytes
      && canvas
      && 'toBlob' in canvas
      && typeof canvas.toBlob === 'function'
      && typeof Promise === 'function'
    ) {
      const encodedBlob = await new Promise((resolve) => {
        canvas.toBlob((value) => {
          resolve(value || null);
        }, 'image/png');
      });

      if (encodedBlob) {
        encodedBytes = new Uint8Array(await encodedBlob.arrayBuffer());
      }
    }

    if (!encodedBytes || encodedBytes.length === 0) {
      return null;
    }

    return {
      bytes: encodedBytes,
      width,
      height,
      sourceWidth,
      sourceHeight,
      resized: width !== sourceWidth || height !== sourceHeight,
    };
  } catch (_) {
    return null;
  } finally {
    try {
      bitmap?.close?.();
    } catch (_) {
      // ignore best-effort bitmap cleanup failures
    }
  }
}
