graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 487
  arrival_time 8961.787438959189
  lifetime 1145.8991670621356
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 11
    gpu 23
    rom 28
  ]
  node [
    id 1
    label "1"
    cpu 9
    gpu 37
    rom 6
  ]
  node [
    id 2
    label "2"
    cpu 47
    gpu 20
    rom 37
  ]
  node [
    id 3
    label "3"
    cpu 8
    gpu 0
    rom 2
  ]
  node [
    id 4
    label "4"
    cpu 1
    gpu 17
    rom 37
  ]
  node [
    id 5
    label "5"
    cpu 24
    gpu 29
    rom 22
  ]
  node [
    id 6
    label "6"
    cpu 4
    gpu 21
    rom 24
  ]
  node [
    id 7
    label "7"
    cpu 20
    gpu 24
    rom 27
  ]
  node [
    id 8
    label "8"
    cpu 21
    gpu 9
    rom 31
  ]
  node [
    id 9
    label "9"
    cpu 14
    gpu 21
    rom 36
  ]
  node [
    id 10
    label "10"
    cpu 36
    gpu 23
    rom 25
  ]
  node [
    id 11
    label "11"
    cpu 33
    gpu 20
    rom 26
  ]
  node [
    id 12
    label "12"
    cpu 5
    gpu 12
    rom 35
  ]
  node [
    id 13
    label "13"
    cpu 42
    gpu 1
    rom 49
  ]
  node [
    id 14
    label "14"
    cpu 5
    gpu 6
    rom 44
  ]
  edge [
    source 0
    target 1
    bw 45
  ]
  edge [
    source 1
    target 2
    bw 38
  ]
  edge [
    source 2
    target 3
    bw 5
  ]
  edge [
    source 3
    target 4
    bw 34
  ]
  edge [
    source 4
    target 5
    bw 45
  ]
  edge [
    source 5
    target 6
    bw 31
  ]
  edge [
    source 6
    target 7
    bw 28
  ]
  edge [
    source 7
    target 8
    bw 0
  ]
  edge [
    source 8
    target 9
    bw 49
  ]
  edge [
    source 9
    target 10
    bw 11
  ]
  edge [
    source 10
    target 11
    bw 21
  ]
  edge [
    source 11
    target 12
    bw 47
  ]
  edge [
    source 12
    target 13
    bw 34
  ]
  edge [
    source 13
    target 14
    bw 3
  ]
]
