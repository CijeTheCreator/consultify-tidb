import { NextRequest, NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const consultationId = params.id
    
    let consultation = await prisma.consultation.findUnique({
      where: {
        consultation_id: consultationId
      }
    })

    // If consultation doesn't exist, create it with default TRIAGE state
    if (!consultation) {
      consultation = await prisma.consultation.create({
        data: {
          consultation_id: consultationId,
          patient_id: "patient_123",
          doctor_id: null,
          state: "TRIAGE",
          patient_language: "fr",
          doctor_language: "en"
        }
      })
    }

    // Transform to match the expected format
    const consultationDetails = {
      patient_id: consultation.patient_id,
      doctor_id: consultation.doctor_id,
      state: consultation.state,
      created_at: consultation.created_at.toISOString(),
      patient_language: consultation.patient_language,
      doctor_language: consultation.doctor_language
    }

    return NextResponse.json(consultationDetails)
  } catch (error) {
    console.error('Error fetching consultation:', error)
    return NextResponse.json({ error: 'Failed to fetch consultation' }, { status: 500 })
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const consultationId = params.id
    const body = await request.json()
    const { state, doctor_id } = body

    const consultation = await prisma.consultation.upsert({
      where: {
        consultation_id: consultationId
      },
      update: {
        state,
        doctor_id,
        updated_at: new Date()
      },
      create: {
        consultation_id: consultationId,
        patient_id: "patient_123",
        doctor_id,
        state,
        patient_language: "fr",
        doctor_language: "en"
      }
    })

    // Transform to match the expected format
    const consultationDetails = {
      patient_id: consultation.patient_id,
      doctor_id: consultation.doctor_id,
      state: consultation.state,
      created_at: consultation.created_at.toISOString(),
      patient_language: consultation.patient_language,
      doctor_language: consultation.doctor_language
    }

    return NextResponse.json(consultationDetails)
  } catch (error) {
    console.error('Error updating consultation:', error)
    return NextResponse.json({ error: 'Failed to update consultation' }, { status: 500 })
  }
}