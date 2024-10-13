package com.example.fuck

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image.Plane
import android.os.Build
import androidx.annotation.RequiresApi
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

object BitmapUtils {
    /**
     * Converts NV21 format byte buffer to bitmap.
     */
    fun getBitmap(data: ByteBuffer?, metadata: FrameMetadata): Bitmap? {
        try {
            data?.rewind()
            val imageInBuffer = data?.let { ByteArray(it.limit()) }
            if (imageInBuffer != null) {
                data.get(imageInBuffer, 0, imageInBuffer.size)
            }
            var baseQuality: String? = "1"
            var eKycPhotoCompression = 1f
            if (!baseQuality.isNullOrEmpty()) {
                eKycPhotoCompression = baseQuality.toFloat() //its already used as float in IOS app, keep it as is
            }
            val image = YuvImage(
                    imageInBuffer, ImageFormat.NV21, metadata.width, metadata.height, null)
            val stream = ByteArrayOutputStream()
            image.compressToJpeg(Rect(0, 0, metadata.width, metadata.height), (eKycPhotoCompression * 100).toInt(), stream)
            val bmp = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size())
            stream.close()
            return rotateBitmap(bmp, metadata.rotation, false, false)
        } catch (e: Exception) {
            
        }
        return null
    }

    @RequiresApi(Build.VERSION_CODES.LOLLIPOP)
    @ExperimentalGetImage
    fun getBitmap(image: ImageProxy): Bitmap? {
        return try {
            val frameMetadata = FrameMetadata.Builder()
                    .setWidth(image.width)
                    .setHeight(image.height)
                    .setRotation(image.imageInfo.rotationDegrees)
                    .build()
            val nv21Buffer = yuv420ThreePlanesToNV21(image.image?.planes?: emptyArray(), image.width, image.height)
            getBitmap(nv21Buffer, frameMetadata)
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Rotates a bitmap if it is converted from a bytebuffer.
     */
    private fun rotateBitmap(
            bitmap: Bitmap, rotationDegrees: Int, flipX: Boolean, flipY: Boolean): Bitmap? {
        return try {
            val matrix = Matrix()
            // Rotate the image back to straight.
            matrix.postRotate(rotationDegrees.toFloat())
            // Mirror the image along the X or Y axis.
            matrix.postScale(if (flipX) -1.0f else 1.0f, if (flipY) -1.0f else 1.0f)
            val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            // Recycle the old bitmap if it has changed.
            if (rotatedBitmap != bitmap) {
                bitmap.recycle()
            }
            rotatedBitmap
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Converts YUV_420_888 to NV21 bytebuffer.
     *
     *
     * The NV21 format consists of a single byte array containing the Y, U and V values. For an
     * image of size S, the first S positions of the array contain all the Y values. The remaining
     * positions contain interleaved V and U values. U and V are subsampled by a factor of 2 in both
     * dimensions, so there are S/4 U values and S/4 V values. In summary, the NV21 array will contain
     * S Y values followed by S/4 VU values: YYYYYYYYYYYYYY(...)YVUVUVUVU(...)VU
     *
     *
     * YUV_420_888 is a generic format that can describe any YUV image where U and V are subsampled
     * by a factor of 2 in both dimensions. [Image.getPlanes] returns an array with the Y, U and
     * V planes. The Y plane is guaranteed not to be interleaved, so we can just copy its values into
     * the first part of the NV21 array. The U and V planes may already have the representation in the
     * NV21 format. This happens if the planes share the same buffer, the V buffer is one position
     * before the U buffer and the planes have a pixelStride of 2. If this is case, we can just copy
     * them to the NV21 array.
     */
    private fun yuv420ThreePlanesToNV21(
            yuv420888planes: Array<Plane>, width: Int, height: Int): ByteBuffer? {
        return try {
            val imageSize = width * height
            val out = ByteArray(imageSize + 2 * (imageSize / 4))
            if (areUVPlanesNV21(yuv420888planes, width, height)) {
                // Copy the Y values.
                yuv420888planes[0].buffer[out, 0, imageSize]
                val uBuffer = yuv420888planes[1].buffer
                val vBuffer = yuv420888planes[2].buffer
                // Get the first V value from the V buffer, since the U buffer does not contain it.
                vBuffer[out, imageSize, 1]
                // Copy the first U value and the remaining VU values from the U buffer.
                uBuffer[out, imageSize + 1, 2 * imageSize / 4 - 1]
            } else {
                // Fallback to copying the UV values one by one, which is slower but also works.
                // Unpack Y.
                unpackPlane(yuv420888planes[0], width, height, out, 0, 1)
                // Unpack U.
                unpackPlane(yuv420888planes[1], width, height, out, imageSize + 1, 2)
                // Unpack V.
                unpackPlane(yuv420888planes[2], width, height, out, imageSize, 2)
            }
            ByteBuffer.wrap(out)
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Checks if the UV plane buffers of a YUV_420_888 image are in the NV21 format.
     */
    private fun areUVPlanesNV21(planes: Array<Plane>, width: Int, height: Int): Boolean {
        return try {
            val imageSize = width * height
            val uBuffer = planes[1].buffer
            val vBuffer = planes[2].buffer

            // Backup buffer properties.
            val vBufferPosition = vBuffer.position()
            val uBufferLimit = uBuffer.limit()

            // Advance the V buffer by 1 byte, since the U buffer will not contain the first V value.
            vBuffer.position(vBufferPosition + 1)
            // Chop off the last byte of the U buffer, since the V buffer will not contain the last U value.
            uBuffer.limit(uBufferLimit - 1)

            // Check that the buffers are equal and have the expected number of elements.
            val areNV21 = vBuffer.remaining() == 2 * imageSize / 4 - 2 && vBuffer.compareTo(uBuffer) == 0

            // Restore buffers to their initial state.
            vBuffer.position(vBufferPosition)
            uBuffer.limit(uBufferLimit)
            areNV21
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Unpack an image plane into a byte array.
     *
     *
     * The input plane data will be copied in 'out', starting at 'offset' and every pixel will be
     * spaced by 'pixelStride'. Note that there is no row padding on the output.
     */
    private fun unpackPlane(
            plane: Plane, width: Int, height: Int, out: ByteArray, offset: Int, pixelStride: Int) {
        try {
            val buffer = plane.buffer
            buffer.rewind()

            // Compute the size of the current plane.
            // We assume that it has the aspect ratio as the original image.
            val numRow = (buffer.limit() + plane.rowStride - 1) / plane.rowStride
            if (numRow == 0) {
                return
            }
            val scaleFactor = height / numRow
            val numCol = width / scaleFactor

            // Extract the data in the output buffer.
            var outputPos = offset
            var rowStart = 0
            for (row in 0 until numRow) {
                var inputPos = rowStart
                for (col in 0 until numCol) {
                    out[outputPos] = buffer[inputPos]
                    outputPos += pixelStride
                    inputPos += plane.pixelStride
                }
                rowStart += plane.rowStride
            }
        } catch (e: Exception) {
            
        }
    }
}