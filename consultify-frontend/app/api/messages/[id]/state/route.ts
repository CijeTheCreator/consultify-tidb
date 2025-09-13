import { NextRequest, NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function PATCH(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const messageId = params.id
    const { state } = await request.json()

    if (state === undefined) {
      return NextResponse.json(
        { error: 'State is required' },
        { status: 400 }
      )
    }

    const message = await prisma.message.findUnique({
      where: { id: messageId },
    })

    if (!message) {
      return NextResponse.json(
        { error: 'Message not found' },
        { status: 404 }
      )
    }

    const updatedMessage = await prisma.message.update({
      where: { id: messageId },
      data: {
        state: state,
      },
      include: {
        sender: true,
        consultation: true,
      },
    })

    return NextResponse.json(updatedMessage)
  } catch (error) {
    console.error('Error updating message state:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}