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
        prescriptionAssistanceState: true,
      },
    })

    if (!consultation) {
      return NextResponse.json(
        { error: 'Consultation not found' },
        { status: 404 }
      )
    }

    return NextResponse.json({ prescriptionAssistanceState: consultation.prescriptionAssistanceState })
  } catch (error) {
    console.error('Error fetching prescription assistance state:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const consultationId = params.id
    const { prescriptionAssistanceState } = await request.json()

    if (!prescriptionAssistanceState) {
      return NextResponse.json(
        { error: 'Prescription assistance state is required' },
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
        prescriptionAssistanceState,
      },
      include: {
        patient: true,
        doctor: true,
      },
    })

    return NextResponse.json(updatedConsultation)
  } catch (error) {
    console.error('Error updating prescription assistance state:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}