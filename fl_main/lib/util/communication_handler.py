# communication_handler.py


import websockets
"""
WebSockets are a communication protocol that allows two-way, real-time communication between a client (like a browser or mobile app) and a server, over a single TCP connection.
Unlike HTTP (which is request-response only):
With HTTP → the client asks, the server responds, and the connection closes.
With WebSockets → once the connection is established, both client and server can send messages anytime (full-duplex communication).
"""
import asyncio
import pickle
import logging



def init_db_server(func, ip, socket):
    """
    Start the DB server
    :param func: Function
    :param ip: IP address
    :param socket: port num
    :return: 
    """
    start_server = websockets.serve(func, ip, socket,
                                    max_size=None, max_queue=None)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()

def init_fl_server(register, receive_msg_from_agent, model_synthesis_routine, aggr_ip, reg_socket, recv_socket):
    """
    Start the FL server on aggregator side
    :param register: Function
    :param receive_msg_from_agent: Function
    :param model_synthesis_routine: Function
    :param aggr_ip: IP address
    :param reg_socket: port num
    :param recv_socket: port num
    :return: 
    """
    loop = asyncio.get_event_loop() #Concurrency in I/O-bound tasks
    start_server = websockets.serve(register, aggr_ip, reg_socket,
                                    max_size=None, max_queue=None)
    start_receiver = websockets.serve(receive_msg_from_agent, aggr_ip, recv_socket,
                                      max_size=None, max_queue=None)
    loop.run_until_complete(asyncio.gather(start_server,
                                           start_receiver,
                                           model_synthesis_routine))
    loop.run_forever()

def init_client_server(func, ip, socket):
    """
    Start the client server
    :param func: Function
    :param ip: IP address
    :param socket: port num
    :return: 
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client_server = websockets.serve(func, ip, socket, max_size=None, max_queue=None)
    loop.run_until_complete(asyncio.gather(client_server))
    loop.run_forever()

async def send(msg, ip, socket):
    """
    Send a message to the IP address and socket
    :param ip: IP address
    :param socket: port num
    :return: response message
    """
    resp = None
    try:
        print("\n entered in try block on send function \n")
        wsaddr = f'ws://{ip}:{socket}'
        async with websockets.connect(wsaddr, max_size=None, max_queue=None, ping_interval=None) as websocket:
            print("\nmsg : ",msg)
            await websocket.send(pickle.dumps(msg))
            try:
                print("\n entered in nested try block")
                rmsg = await websocket.recv()
                print(f" rmsg : {rmsg}")
                resp = pickle.loads(rmsg)
            except:
                # logging.info("--- Nothing to be received ---")
                print("\n entered inner nested except block ")
                pass

            return resp
    except:
        print("\n entered to outer except block  ")
        logging.error("Connection lost to the agent: " + ip)
        logging.error(f'--- Message NOT Sent ---')
        return resp

async def send_websocket(msg, websocket):
    """
    Send a binary file (message) to an agent through a give websocket
    :param bsgms: Message (binary file)
    :param websocket:
    :return:
    """
    while not websocket:  # wait until socket being initialized
        await asyncio.sleep(0.001)
    await websocket.send(pickle.dumps(msg))

async def receive(websocket):
    """
    Receive the message from the websocket
    :param websocket:
    :return: A pickled message
    """
    return pickle.loads(await websocket.recv())
    