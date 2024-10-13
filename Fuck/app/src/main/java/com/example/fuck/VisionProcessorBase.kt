package com.example.fuck

import android.graphics.Bitmap
import android.os.Build.VERSION_CODES
import androidx.annotation.GuardedBy
import androidx.annotation.RequiresApi
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskExecutors
import com.google.mlkit.vision.common.InputImage
import java.nio.ByteBuffer

abstract class VisionProcessorBase<T> : VisionImageProcessor {

    // To keep the latest images and its metadata.
    @GuardedBy("this")
    private var latestImage: ByteBuffer? = null

    @GuardedBy("this")
    private var latestImageMetaData: FrameMetadata? = null

    // To keep the images and metadata in process.
    @GuardedBy("this")
    private var processingImage: ByteBuffer? = null

    @GuardedBy("this")
    private var processingMetaData: FrameMetadata? = null
    private val executor = ScopedExecutor(TaskExecutors.MAIN_THREAD)


    @Synchronized
    override fun process(
        data: ByteBuffer?,
        frameMetadata: FrameMetadata?,
        graphicOverlay: OverlayView?) {
        try {
            latestImage = data
            latestImageMetaData = frameMetadata
            if (processingImage == null && processingMetaData == null && graphicOverlay!=null) {
                processLatestImage(graphicOverlay)
            }
        } catch (e: Exception) {
        }
    }


    // Bitmap version
    override fun process(bitmap: Bitmap?, graphicOverlay: OverlayView?) {
        if (bitmap != null && graphicOverlay != null) {
            detectInVisionImage(
                null, /* bitmap */
                InputImage.fromBitmap(bitmap, 0),
                null,
                graphicOverlay)
        }
    }

    // -----------------Code for processing live preview frame from CameraX API-----------------------
    @RequiresApi(VERSION_CODES.LOLLIPOP)
    @ExperimentalGetImage
    override fun processImageProxy(image: ImageProxy?, graphicOverlay: OverlayView?) {
        try {
            if (image != null && graphicOverlay != null) {
                var bitmap = BitmapUtils.getBitmap(image)
                requestDetectInImage(
                    InputImage.fromMediaImage(image.image!!, image.imageInfo.rotationDegrees),
                    graphicOverlay,
                    bitmap)
                    .addOnCompleteListener { image.close() }

            }
        } catch (e: Exception) {
        }

    }

    // -----------------Common processing logic-------------------------------------------------------
    private fun requestDetectInImage(image: InputImage, graphicOverlay: OverlayView, originalCameraImage: Bitmap?): Task<T> {
        return setUpListener(
            detectInImage(image),
            graphicOverlay,
            originalCameraImage
        )
    }

    private fun setUpListener(
        task: Task<T>,
        graphicOverlay: OverlayView,
        originalCameraImage: Bitmap?,
    ): Task<T> {
        return task
            .addOnSuccessListener(executor
            ) { results: T ->
                onSuccess(originalCameraImage, results, graphicOverlay)
            }
            .addOnFailureListener(
                executor
            ) { e: Exception ->
                graphicOverlay.clear()
                graphicOverlay.postInvalidate()
                val error = "Failed to process. Error: " + e.localizedMessage
                e.printStackTrace()
                this@VisionProcessorBase.onFailure(e)
            }
    }


    @Synchronized
    private fun processLatestImage(graphicOverlay: OverlayView) {
        try {
            processingImage = latestImage
            processingMetaData = latestImageMetaData
            latestImage = null
            latestImageMetaData = null
            if (processingImage != null && processingMetaData != null) {
                processImage(processingImage!!, processingMetaData!!, graphicOverlay)
            }
        } catch (e: Exception) {
        }
    }

    override fun processByteBuffer(data: ByteBuffer?, frameMetadata: FrameMetadata?, graphicOverlay: OverlayView?) {
        try {
            latestImage = data
            latestImageMetaData = frameMetadata
            if (processingImage == null && processingMetaData == null) {
                processLatestImage(graphicOverlay!!)
            }
        } catch (e: Exception) {
        }
    }

    private fun processImage(
        data: ByteBuffer,
        frameMetadata: FrameMetadata,
        graphicOverlay: OverlayView
    ) {
        try {
            val bitmap = BitmapUtils.getBitmap(data, frameMetadata)
            detectInVisionImage(
                bitmap, InputImage.fromByteBuffer(
                    data,
                    frameMetadata.width,
                    frameMetadata.height,
                    frameMetadata.rotation,
                    InputImage.IMAGE_FORMAT_NV21
                ), frameMetadata, graphicOverlay)
        } catch (e: Exception) {
        }
    }

    private fun detectInVisionImage(
        originalCameraImage: Bitmap?,
        image: InputImage,
        metadata: FrameMetadata?,
        graphicOverlay: OverlayView
    ) {
        try {
            detectInImage(image)
                .addOnSuccessListener { results ->
                    onSuccess(originalCameraImage, results, graphicOverlay)
                    processLatestImage(graphicOverlay)
                }
                .addOnFailureListener { e -> onFailure(e) }
        } catch (e: Exception) {
        }
    }

    override fun stop() {}

    protected abstract fun detectInImage(image: InputImage): Task<T>

    /**
     * Callback that executes with a successful detection result.
     *
     * @param originalCameraImage hold the original image from camera, used to draw the background
     * image.
     */
    protected abstract fun onSuccess(
        originalCameraImage: Bitmap?,
        results: T,
        graphicOverlay: OverlayView
    )

    protected abstract fun onFailure(e: Exception)
}