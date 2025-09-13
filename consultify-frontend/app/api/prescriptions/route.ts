import { NextRequest, NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function POST(request: NextRequest) {
  try {
    const { 
      drugName, 
      frequency, 
      startTimestamp, 
      endTimestamp, 
      patientId, 
      consultationId 
    } = await request.json()

    if (!drugName || !frequency || !startTimestamp || !endTimestamp || !patientId || !consultationId) {
      return NextResponse.json(
        { error: 'All fields are required' },
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

    const prescription = await prisma.prescription.create({
      data: {
        drugName,
        frequency,
        startTimestamp: new Date(startTimestamp),
        endTimestamp: new Date(endTimestamp),
        patientId,
        consultationId,
      },
      include: {
        patient: true,
        consultation: true,
      },
    })

    return NextResponse.json(prescription)
  } catch (error) {
    console.error('Error creating prescription:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const patientId = searchParams.get('patientId')
    const consultationId = searchParams.get('consultationId')

    let where: any = {}

    if (patientId) {
      where.patientId = patientId
    }

    if (consultationId) {
      where.consultationId = consultationId
    }

    const prescriptions = await prisma.prescription.findMany({
      where,
      include: {
        patient: true,
        consultation: true,
      },
      orderBy: { createdAt: 'desc' },
    })

    return NextResponse.json(prescriptions)
  } catch (error) {
    console.error('Error fetching prescriptions:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}