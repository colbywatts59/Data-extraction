package com.example.gridbuttonapp.network

import com.example.gridbuttonapp.data.RecordingRequest
import com.example.gridbuttonapp.data.RecordingResponse
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

/**
 * Retrofit API interface for recording endpoints
 */
interface RecordingApi {
    @POST("touch/initialize")
    suspend fun initialize(): Response<RecordingResponse>
    
    @POST("touch/start")
    suspend fun startRecording(@Body request: RecordingRequest): Response<RecordingResponse>
    
    @POST("touch/ping")
    suspend fun ping(): Response<RecordingResponse>
    
    @POST("touch/toggle_graphs")
    suspend fun toggleGraphs(@Body request: Map<String, Boolean>): Response<RecordingResponse>
    
    @POST("touch/set_dataset_mode")
    suspend fun setDatasetMode(@Body request: Map<String, String>): Response<RecordingResponse>
}

