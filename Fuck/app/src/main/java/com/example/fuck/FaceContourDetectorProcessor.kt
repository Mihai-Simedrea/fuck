package com.example.fuck

import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.IOException

/**
 * Face Contour Demo.
 */
class FaceContourDetectorProcessor(
    faceContourDetectorListener: FaceContourDetectorListener? = null,
    isShowDot: Boolean = false
) : VisionProcessorBase<List<Face>>() {


    private val detector: FaceDetector
    private var mFaceContourDetectorListener: FaceContourDetectorListener? = null
    private var rotationMax = 12
    private var rotationMin = -12
    private var eye = 0.5f
    private var smiling = 0.3f
    private var isMlkit = true
    var left = 0F
    var right = 0F
    var top = 0F
    var bottom = 0F

    init {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setContourMode(if (isShowDot) FaceDetectorOptions.CONTOUR_MODE_ALL else FaceDetectorOptions.CONTOUR_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(0.5f)
            .build()

        rotationMax = 12
        rotationMin = -12
        eye = 0.5F
        smiling = 0.0F


        detector = FaceDetection.getClient(options)
        mFaceContourDetectorListener = faceContourDetectorListener
    }

    override fun stop() {
        try {
            detector.close()
        } catch (e: IOException) {
            Log.e(TAG, "Exception thrown while trying to close Face Contour Detector: $e")
        }
    }

    override fun detectInImage(image: InputImage): Task<List<Face>> {
        return detector.process(image)
    }

    override fun onSuccess(
        originalCameraImage: Bitmap?,
        faces: List<Face>,
        graphicOverlay: OverlayView
    ) {
        try {
            graphicOverlay.clear()
            if(isFaceInsideRectangle(faces,graphicOverlay)) {
                if (faces.isEmpty()) {
                    mFaceContourDetectorListener?.onNoFaceDetected()
                } else {
                    if (isMlkit) {
                        for (face in faces) {
                            val rotY = face.headEulerAngleY // Head is rotated to the right rotY degrees
                            val faceSmiling = face.smilingProbability
                            if (rotY <= rotationMax && rotY >= rotationMin && (face.leftEyeOpenProbability
                                    ?: 0.0f) > eye && (face.rightEyeOpenProbability
                                    ?: 0.0f) > eye && (faceSmiling
                                    ?: 0.0f) > smiling
                            ) {
                                originalCameraImage?.let {
                                    Log.e("left1",""+face.boundingBox.left)
                                    Log.e("left2",""+graphicOverlay.rectF.left)
                                    Log.e("top1",""+face.boundingBox.top)
                                    Log.e("top2",""+graphicOverlay.rectF.top)
                                    Log.e("right1",""+face.boundingBox.right)
                                    Log.e("right2",""+graphicOverlay.rectF.right)
                                    Log.e("bottom1",""+face.boundingBox.bottom)
                                    Log.e("bottom2",""+graphicOverlay.rectF.bottom)
                                    var bitmap = extractFace(it,face.boundingBox.left.toInt(),face.boundingBox.top.toInt(),face.boundingBox.width(),face.boundingBox.height())
                                    val imageWidth: Int = bitmap?.width?:0
                                    val imageHeight: Int = bitmap?.height?:0
                                    Log.e("width height", "$imageWidth $imageHeight")
                                    /* var bitmap = extractFace(it,graphicOverlay.rectF.left.toInt(),
                                         graphicOverlay.rectF.top.toInt(),
                                         graphicOverlay.rectF.right.toInt()-graphicOverlay.rectF.left.toInt(),
                                         graphicOverlay.rectF.bottom.toInt()-graphicOverlay.rectF.top.toInt())*/
                                    bitmap?.let {
                                        mFaceContourDetectorListener?.onCapturedFace(it)
                                    }
                                }
                            } else {
                                mFaceContourDetectorListener?.onNoFaceDetected()
                            }
                        }
                    } else {
                        originalCameraImage?.let { mFaceContourDetectorListener?.onCapturedFace(it) }
                    }
                }
            }else{
                mFaceContourDetectorListener?.onNoFaceDetected()
            }
            graphicOverlay.postInvalidate()
        } catch (e: Exception) {
            Log.e("Exception",""+e.localizedMessage)
        }
    }

    override fun onFailure(e: Exception) {
        try {
            Log.e(TAG, "Face detection failed ${e.message}")
        } catch (e: Exception) {
        }
    }

    private fun extractFace(bmp: Bitmap, x: Int, y: Int, width: Int, height: Int): Bitmap? {
        val originX = if (x + width > bmp.width) (bmp.width - width) else x
        val originY = if (y + height > bmp.height) (bmp.height - height) else y
        return Bitmap.createBitmap(bmp, originX-80, originY-50, width+150, height+150)
    }

    fun translateX(x: Float,overlay: OverlayView): Float {
        return if (overlay.isImageFlipped) {
            overlay.width - (scale(x,overlay) - overlay.postScaleWidthOffset)
        } else {
            x - overlay.postScaleWidthOffset
        }
    }

    fun isFaceInsideRectangle(faces: List<Face>, graphicOverlay: OverlayView):Boolean{
        try {
            faces.forEach { face ->
                val x = translateX(face.boundingBox.centerX().toFloat(),graphicOverlay)
                val y = translateY(face.boundingBox.centerY().toFloat(), graphicOverlay)

                // Draws a bounding box around the face.
                left = x - scale(face.boundingBox.width() / 2.0f, graphicOverlay)
                top = y - scale(face.boundingBox.height() / 2.0f, graphicOverlay)
                right = x + scale(face.boundingBox.width() / 2.0f, graphicOverlay)
                bottom = y + scale(face.boundingBox.height() / 2.0f, graphicOverlay)
            }
            var isFaceInsideRectangle = faces.any { left > graphicOverlay.rectF.left &&
                    top > graphicOverlay.rectF.top &&
                    bottom < graphicOverlay.rectF.bottom &&
                    right < graphicOverlay.rectF.right
            }
            return isFaceInsideRectangle
        }catch (e:Exception){
            return false
        }
    }

    /**
     * Adjusts the y coordinate from the image's coordinate system to the view coordinate system.
     */
    fun translateY(y: Float,overlay:OverlayView): Float {
        return scale(y,overlay) - overlay.postScaleHeightOffset
    }

    fun scale(imagePixel: Float,overlay:OverlayView): Float {
        return imagePixel * overlay.scaleFactor
    }



    companion object {
        private const val TAG = "FaceContourDetectorProc"
    }

    interface FaceContourDetectorListener {
        fun onCapturedFace(originalCameraImage: Bitmap)
        fun onNoFaceDetected()
    }
}