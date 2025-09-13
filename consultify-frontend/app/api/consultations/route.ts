import { NextRequest, NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function POST(request: NextRequest) {
  try {
    const { patientId } = await request.json()

    if (!patientId) {
      return NextResponse.json(
        { error: 'Patient ID is required' },
        { status: 400 }
      )
    }

    const patient = await prisma.user.findUnique({
      where: { id: patientId, type: 'PATIENT' },
    })

    if (!patient) {
      return NextResponse.json(
        { error: 'Patient not found' },
        { status: 404 }
      )
    }

    const consultation = await prisma.consultation.create({
      data: {
        patientId,
        state: 'CLERKING',
      },
      include: {
        patient: true,
        doctor: true,
      },
    })

    return NextResponse.json(consultation)
  } catch (error) {
    console.error('Error creating consultation:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')
    const userId = searchParams.get('userId')
    const userType = searchParams.get('userType')

    // If consultation ID is provided, get single consultation
    if (id) {
      const consultation = await prisma.consultation.findUnique({
        where: { id },
        include: {
          patient: true,
          doctor: true,
          prescriptions: true,
        },
      })

      if (!consultation) {
        return NextResponse.json(
          { error: 'Consultation not found' },
          { status: 404 }
        )
      }

      return NextResponse.json(consultation)
    }

    // If userId and userType are provided, get all consultations for user
    if (userId && userType) {
      let consultations;
      
      if (userType === 'PATIENT') {
        consultations = await prisma.consultation.findMany({
          where: { patientId: userId },
          include: {
            patient: true,
            doctor: true,
            prescriptions: true,
          },
          orderBy: { createdAt: 'desc' }
        })
      } else if (userType === 'DOCTOR') {
        consultations = await prisma.consultation.findMany({
          where: { doctorId: userId },
          include: {
            patient: true,
            doctor: true,
            prescriptions: true,
          },
          orderBy: { createdAt: 'desc' }
        })
      } else {
        return NextResponse.json(
          { error: 'Invalid user type' },
          { status: 400 }
        )
      }

      return NextResponse.json(consultations)
    }

    return NextResponse.json(
      { error: 'Either consultation ID or userId with userType is required' },
      { status: 400 }
    )
  } catch (error) {
    console.error('Error fetching consultations:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}