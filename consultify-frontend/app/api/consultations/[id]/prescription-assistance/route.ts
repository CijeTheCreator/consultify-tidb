import { NextRequest, NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const consultationId = params.id

    const consultation = await prisma.consultation.findUnique({
      where: { id: consultationId },
      select: {
        id: true,
        prescriptionAssistance: true,
      },
    })

    if (!consultation) {
      return NextResponse.json(
        { error: 'Consultation not found' },
        { status: 404 }
      )
    }

    return NextResponse.json({ prescriptionAssistance: consultation.prescriptionAssistance })
  } catch (error) {
    console.error('Error fetching prescription assistance:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const consultationId = params.id
    const { prescriptionAssistance } = await request.json()

    if (!prescriptionAssistance) {
      return NextResponse.json(
        { error: 'Prescription assistance content is required' },
        { status: 400 }
      )
    }

    const consultation = await prisma.consultation.findUnique({
      where: { id: consultationId },
    })

    if (!consultation) {
      return NextResponse.json(
        { error: 'Consultation not found' },
        { status: 404 }
      )
    }

    const updatedConsultation = await prisma.consultation.update({
      where: { id: consultationId },
      data: {
        prescriptionAssistance,
      },
      include: {
        patient: true,
        doctor: true,
      },
    })

    return NextResponse.json(updatedConsultation)
  } catch (error) {
    console.error('Error creating prescription assistance:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}