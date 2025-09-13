import { NextRequest, NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function PATCH(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const consultationId = params.id
    const { doctorId } = await request.json()

    if (!doctorId) {
      return NextResponse.json(
        { error: 'Doctor ID is required' },
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

    const doctor = await prisma.user.findUnique({
      where: { id: doctorId, type: 'DOCTOR' },
    })

    if (!doctor) {
      return NextResponse.json(
        { error: 'Doctor not found' },
        { status: 404 }
      )
    }

    const updatedConsultation = await prisma.consultation.update({
      where: { id: consultationId },
      data: {
        doctorId,
        state: 'CONSULTING',
      },
      include: {
        patient: true,
        doctor: true,
      },
    })

    return NextResponse.json(updatedConsultation)
  } catch (error) {
    console.error('Error assigning doctor:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}