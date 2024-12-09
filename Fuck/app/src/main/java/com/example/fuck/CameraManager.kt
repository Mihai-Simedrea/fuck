package com.example.fuck

import android.annotation.SuppressLint
import android.content.Context
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mlkit.common.MlKitException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraManager(
    private val context: Context,
    private val finderView: PreviewView,
    private val lifecycleOwner: LifecycleOwner,
    private val overlayView: OverlayView, faceContourDetectorListener: FaceContourDetectorProcessor.FaceContourDetectorListener) {

    private var preview: Preview? = null
    private lateinit var cameraExecutor: ExecutorService
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalyzer: ImageAnalysis? = null
    var imageProcessor: VisionImageProcessor? = null
    private var needUpdateGraphicOverlayImageSourceInfo = false
    private lateinit var cameraSelector: CameraSelector
    var mFaceContourDetectorListener: FaceContourDetectorProcessor.FaceContourDetectorListener? = null

    init {
        mFaceContourDetectorListener = faceContourDetectorListener
        createNewExecutor()
    }

    private fun createNewExecutor() {
        try {
            cameraExecutor = Executors.newSingleThreadExecutor()
        } catch (e: Exception) {

        }
    }

    fun startCamera() {
        try {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            cameraProviderFuture.addListener(
                {
                    cameraProvider = cameraProviderFuture.get()
                    bindAllCameraUseCases()
                }, ContextCompat.getMainExecutor(context)
            )
        } catch (e: Exception) {

        }
    }


    private fun bindAllCameraUseCases() {
        try {
            if (cameraProvider != null) {
                // As required by CameraX API, unbinds all use cases before trying to re-bind any of them.
                cameraProvider?.unbindAll()
                bindPreviewUseCase()
                bindAnalysisUseCase()
            }
        } catch (e: Exception) {

        }
    }

    private fun bindPreviewUseCase() {
        try {
            if (cameraProvider == null) {
                return
            }
            if (preview != null) {
                cameraProvider?.unbind(preview)
            }
            val builder = Preview.Builder()
            cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
            preview = builder.build()
            preview?.setSurfaceProvider(finderView.surfaceProvider)
            //cameraProvider?.bindToLifecycle(lifecycleOwner, cameraSelector, preview)
        } catch (e: Exception) {

        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindAnalysisUseCase() {
        try {
            if (cameraProvider == null) {
                return
            }
            if (imageAnalyzer != null) {
                cameraProvider?.unbind(imageAnalyzer)
            }
            if (imageProcessor != null) {
                imageProcessor?.stop()
            }
            imageProcessor = FaceContourDetectorProcessor(mFaceContourDetectorListener)
            val builder = ImageAnalysis.Builder()
            imageAnalyzer = builder.setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
            needUpdateGraphicOverlayImageSourceInfo = true
            imageAnalyzer?.setAnalyzer(
                // imageProcessor.processImageProxy will use another thread to run the detection underneath,
                // thus we can just runs the analyzer itself on main thread.
                ContextCompat.getMainExecutor(context)
            ) { imageProxy: ImageProxy ->
                try {
                    if (needUpdateGraphicOverlayImageSourceInfo) {
                        val isImageFlipped = lensFacing == CameraSelector.LENS_FACING_FRONT
                        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                        if (rotationDegrees == 0 || rotationDegrees == 180) {
                            overlayView.setImageSourceInfo(imageProxy.width, imageProxy.height, isImageFlipped)
                        } else {
                            overlayView.setImageSourceInfo(imageProxy.height, imageProxy.width, isImageFlipped)
                        }
                        needUpdateGraphicOverlayImageSourceInfo = false
                    }
                    try {
                        imageProcessor?.processImageProxy(imageProxy, overlayView)
                    } catch (e: MlKitException) {
                        Log.e(TAG, "Failed to process image. Error: " + e.localizedMessage)

                    }
                } catch (e: Exception) {

                }
            }
            cameraProvider?.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalyzer)
        } catch (e: Exception) {

        }
    }

    fun stopCamera(){
        try {
            if (cameraProvider != null) {
                cameraProvider?.unbindAll()
            }

            if (imageAnalyzer != null) {
                cameraProvider?.unbind(imageAnalyzer)
            }

            if (imageAnalyzer != null) {
                cameraProvider?.unbind(imageAnalyzer)
            }
            if (imageProcessor != null) {
                imageProcessor?.stop()
            }
        }catch (e:Exception){

        }
    }


    companion object {
        private const val TAG = "CameraXBasic"
    }

}