import { NextRequest, NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const consultationId = params.id
    
    const messages = await prisma.message.findMany({
      where: {
        consultation_id: consultationId
      },
      orderBy: {
        timestamp: 'asc'
      }
    })

    // Transform to match the expected format
    const formattedMessages = messages.map(msg => ({
      message_id: msg.message_id,
      sender: msg.sender,
      timestamp: msg.timestamp,
      translatedContent: msg.translatedContent,
      translatedLanguage: msg.translatedLanguage,
      originalContent: msg.originalContent,
      originalLanguage: msg.originalLanguage,
      attestation: msg.attestation
    }))

    return NextResponse.json(formattedMessages)
  } catch (error) {
    console.error('Error fetching messages:', error)
    return NextResponse.json({ error: 'Failed to fetch messages' }, { status: 500 })
  }
}