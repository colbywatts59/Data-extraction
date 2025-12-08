package com.example.gridbuttonapp.network

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/**
 * Network configuration and Retrofit setup
 */
object NetworkModule {
    
    // ===== CONFIGURATION - MODIFY THESE VALUES =====
    // Default base URL - can be overridden
    private const val DEFAULT_BASE_URL = "http://10.0.0.92:5150/"
    
    // ================================================
    
    private var currentBaseUrl = DEFAULT_BASE_URL
    
    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }
    
    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.SECONDS)
        .writeTimeout(5, TimeUnit.SECONDS)
        .build()
    
    private var retrofit = Retrofit.Builder()
        .baseUrl(currentBaseUrl)
        .client(okHttpClient)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
    
    var recordingApi: RecordingApi = retrofit.create(RecordingApi::class.java)
        private set
    
    /**
     * Update the base URL and recreate the API
     */
    fun updateBaseUrl(newBaseUrl: String) {
        currentBaseUrl = newBaseUrl
        retrofit = Retrofit.Builder()
            .baseUrl(currentBaseUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
        recordingApi = retrofit.create(RecordingApi::class.java)
    }
}

