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
    })

    if (!consultation) {
      return NextResponse.json(
        { error: 'Consultation not found' },
        { status: 404 }
      )
    }

    const messages = await prisma.message.findMany({
      where: { consultationId },
      include: {
        sender: true,
      },
      orderBy: { createdAt: 'asc' },
    })

    return NextResponse.json(messages)
  } catch (error) {
    console.error('Error fetching messages:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}