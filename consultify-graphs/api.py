import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'consultify-graphs'))

from clerking import graph as clerking_graph
from prescriptionAgent import graph as prescription_graph
from models import AgentState, PrescriptionAgentState, Message, Consultation, Doctor

app = Flask(__name__)

@app.route('/clerking', methods=['POST'])
def invoke_clerking():
    """
    Invoke the clerking graph in background
    
    Expected JSON payload:
    {
        "conversation": [{"id": "...", "original_content": "...", ...}],
        "consultation": {"id": "...", "state": "...", ...},
        "next_message_to_append": {"id": "...", "original_content": "...", ...}
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Parse conversation messages
        conversation = []
        if data.get('conversation'):
            for msg_data in data['conversation']:
                conversation.append(Message(**msg_data))
        
        # Parse consultation
        consultation = None
        if data.get('consultation'):
            consultation = Consultation(**data['consultation'])
        
        # Parse next message to append
        next_message = None
        if data.get('next_message_to_append'):
            next_message = Message(**data['next_message_to_append'])
        
        # Parse last inserted message by user
        last_message = None
        if data.get('last_inserted_message_by_user'):
            last_message = Message(**data['last_inserted_message_by_user'])
        
        # Parse doctor if provided
        doctor = None
        if data.get('doctor'):
            doctor = Doctor(**data['doctor'])
        
        # Create AgentState
        state = AgentState(
            conversation=conversation,
            consultation=consultation,
            next_message_to_append=next_message,
            last_inserted_message_by_user=last_message,
            doctor=doctor,
            query=data.get('query'),
            medical_specialty=data.get('medical_specialty'),
            refined_query=data.get('refined_query'),
            context_retrieved=data.get('context_retrieved'),
            doctor_selection_rationale=data.get('doctor_selection_rationale'),
            medical_consultation_summary=data.get('medical_consultation_summary')
        )
        
        # Start the clerking graph in background
        def run_clerking_graph():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(clerking_graph.ainvoke(state))
            except Exception as e:
                print(f"Background clerking graph error: {e}")
            finally:
                loop.close()
        
        # Start background thread
        thread = threading.Thread(target=run_clerking_graph)
        thread.daemon = True
        thread.start()
        
        # Return immediate confirmation
        return jsonify({
            "success": True,
            "message": "Clerking graph started successfully",
            "consultation_id": consultation.id if consultation else None
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/prescription', methods=['POST'])
def invoke_prescription():
    """
    Invoke the prescription agent graph in background
    
    Expected JSON payload:
    {
        "conversation": [{"id": "...", "original_content": "...", ...}],
        "consultation": {"id": "...", "state": "...", ...}
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Parse conversation messages
        conversation = []
        if data.get('conversation'):
            for msg_data in data['conversation']:
                conversation.append(Message(**msg_data))
        
        # Parse consultation
        consultation = None
        if data.get('consultation'):
            consultation = Consultation(**data['consultation'])
        
        # Create PrescriptionAgentState
        state = PrescriptionAgentState(
            conversation=conversation,
            consultation=consultation,
            query=data.get('query'),
            refined_query=data.get('refined_query'),
            context_retrieved=data.get('context_retrieved'),
            prescriptions_recommended=data.get('prescriptions_recommended', [])
        )
        
        # Start the prescription graph in background
        def run_prescription_graph():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(prescription_graph.ainvoke(state))
            except Exception as e:
                print(f"Background prescription graph error: {e}")
            finally:
                loop.close()
        
        # Start background thread
        thread = threading.Thread(target=run_prescription_graph)
        thread.daemon = True
        thread.start()
        
        # Return immediate confirmation
        return jsonify({
            "success": True,
            "message": "Prescription graph started successfully",
            "consultation_id": consultation.id if consultation else None
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
