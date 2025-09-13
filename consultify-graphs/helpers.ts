// TypeScript interfaces based on models.py schemas

export interface Prescription {
  id?: string;
  drug_name?: string;
  frequency?: string;
  start_timestamp?: string;
  end_timestamp?: string;
  patient_id?: string;
  consultation_id?: string;
  created_at?: string;
  updated_at?: string;
}

export enum ConsultationState {
  CLERKING = "CLERKING",
  CONSULTING = "CONSULTING"
}

export interface Consultation {
  id?: string;
  state?: ConsultationState;
  prescription_assistance?: string;
  prescription_assistance_state?: string;
  patient_id?: string;
  clerk_id?: string;
  doctor_id?: string;
  messages?: Message[];
  prescriptions?: Prescription[];
  created_at?: string;
  updated_at?: string;
}

export interface Doctor {
  id?: string;
  language?: string;
  specialty?: string;
}

export interface Message {
  id?: string;
  translated_content?: string;
  translated_language?: string;
  original_language?: string;
  original_content?: string;
  llm_language?: string;
  llm_content?: string;
  state?: string;
  sender_id?: string;
  consultation_id?: string;
  created_at?: string;
  updated_at?: string;
}

export interface AgentState {
  conversation?: Message[];
  last_inserted_message_by_user?: Message;
  query?: string;
  medical_specialty?: string;
  refined_query?: string;
  context_retrieved?: string;
  consultation?: Consultation;
  next_message_to_append?: Message;
  doctor?: Doctor;
  doctor_selection_rationale?: string;
  medical_consultation_summary?: string;
  messages?: any[];
}

export interface PrescriptionAgentState {
  conversation?: Message[];
  query?: string;
  refined_query?: string;
  context_retrieved?: string;
  consultation?: Consultation;
  prescriptions_recommended?: Prescription[];
}

// API Response interfaces
export interface ApiResponse {
  success: boolean;
  message?: string;
  consultation_id?: string;
  error?: string;
}

// API Helper Functions
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5500';

export async function invokeClerkingGraph(data: AgentState): Promise<ApiResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/clerking`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error invoking clerking graph:', error);
    throw error;
  }
}

export async function invokePrescriptionGraph(data: PrescriptionAgentState): Promise<ApiResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/prescription`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error invoking prescription graph:', error);
    throw error;
  }
}

export async function healthCheck(): Promise<{ status: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error checking health:', error);
    throw error;
  }
}

// Unimplemented getter functions - awaiting implementation
export async function getConversation(consultationId: string): Promise<Message[]> {
  // TODO: Implement conversation getter
  throw new Error('Not implemented');
}

export async function getConsultation(consultationId: string): Promise<Consultation> {
  // TODO: Implement consultation getter
  throw new Error('Not implemented');
}

export async function getNextMessageToAppend(consultationId: string): Promise<Message | null> {
  // TODO: Implement next message to append getter
  throw new Error('Not implemented');
}

export async function getLastInsertedMessageByUser(consultationId: string): Promise<Message | null> {
  // TODO: Implement last inserted message by user getter
  throw new Error('Not implemented');
}
